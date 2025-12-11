#!/usr/bin/env python3
"""Checkpoint transfer protocol over NKN with chunking and verification.

This implements reliable file transfer for trained model checkpoints:
- Chunks large files into manageable pieces (1MB each)
- Sends chunks with sequence numbers and checksums
- Receiver verifies completeness and requests missing chunks
- Sender only discards after receiving OK confirmation
"""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Message types for checkpoint transfer
MSG_CHECKPOINT_START = "checkpoint_start"
MSG_CHECKPOINT_CHUNK = "checkpoint_chunk"
MSG_CHECKPOINT_REQUEST_RETRY = "checkpoint_request_retry"
MSG_CHECKPOINT_COMPLETE = "checkpoint_complete"
MSG_CHECKPOINT_ACK = "checkpoint_ack"

CHUNK_SIZE = 1024 * 1024  # 1MB chunks


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint transfer."""
    checkpoint_id: str  # Unique ID for this checkpoint transfer
    filename: str  # Original filename
    total_size: int  # Total file size in bytes
    num_chunks: int  # Total number of chunks
    file_hash: str  # SHA256 hash of complete file
    iteration: int  # Training iteration number
    timestamp: float  # When checkpoint was created


@dataclass
class CheckpointChunk:
    """A single chunk of checkpoint data."""
    checkpoint_id: str
    chunk_index: int  # 0-indexed
    total_chunks: int
    data: bytes
    chunk_hash: str  # SHA256 hash of this chunk


def create_checkpoint_start_message(
    sequencer,
    metadata: CheckpointMetadata,
) -> Dict[str, Any]:
    """Create message to initiate checkpoint transfer.

    Args:
        sequencer: MessageSequencer for ordering
        metadata: Checkpoint metadata

    Returns:
        Message envelope dict
    """
    from remote_protocol_rl import MessageEnvelope

    payload = {
        "checkpoint_id": metadata.checkpoint_id,
        "filename": metadata.filename,
        "total_size": metadata.total_size,
        "num_chunks": metadata.num_chunks,
        "file_hash": metadata.file_hash,
        "iteration": metadata.iteration,
        "timestamp": metadata.timestamp,
    }

    msg = sequencer.create_message(MSG_CHECKPOINT_START, payload)
    return msg.to_dict()


def create_checkpoint_chunk_message(
    sequencer,
    chunk: CheckpointChunk,
) -> Dict[str, Any]:
    """Create message containing a checkpoint chunk.

    Args:
        sequencer: MessageSequencer for ordering
        chunk: Checkpoint chunk data

    Returns:
        Message envelope dict
    """
    from remote_protocol_rl import MessageEnvelope

    payload = {
        "checkpoint_id": chunk.checkpoint_id,
        "chunk_index": chunk.chunk_index,
        "total_chunks": chunk.total_chunks,
        "data": chunk.data.hex(),  # Hex encode for JSON transport
        "chunk_hash": chunk.chunk_hash,
    }

    msg = sequencer.create_message(MSG_CHECKPOINT_CHUNK, payload)
    return msg.to_dict()


def create_checkpoint_request_retry_message(
    sequencer,
    checkpoint_id: str,
    missing_chunks: List[int],
) -> Dict[str, Any]:
    """Create message requesting retransmission of missing chunks.

    Args:
        sequencer: MessageSequencer for ordering
        checkpoint_id: ID of checkpoint being transferred
        missing_chunks: List of chunk indices that are missing

    Returns:
        Message envelope dict
    """
    from remote_protocol_rl import MessageEnvelope

    payload = {
        "checkpoint_id": checkpoint_id,
        "missing_chunks": missing_chunks,
    }

    msg = sequencer.create_message(MSG_CHECKPOINT_REQUEST_RETRY, payload)
    return msg.to_dict()


def create_checkpoint_complete_message(
    sequencer,
    checkpoint_id: str,
) -> Dict[str, Any]:
    """Create message indicating receiver has all chunks.

    Args:
        sequencer: MessageSequencer for ordering
        checkpoint_id: ID of checkpoint being transferred

    Returns:
        Message envelope dict
    """
    from remote_protocol_rl import MessageEnvelope

    payload = {
        "checkpoint_id": checkpoint_id,
    }

    msg = sequencer.create_message(MSG_CHECKPOINT_COMPLETE, payload)
    return msg.to_dict()


def create_checkpoint_ack_message(
    sequencer,
    checkpoint_id: str,
    success: bool,
    saved_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Create acknowledgment message after checkpoint verification.

    Args:
        sequencer: MessageSequencer for ordering
        checkpoint_id: ID of checkpoint being transferred
        success: Whether checkpoint was successfully verified and saved
        saved_path: Where checkpoint was saved (if successful)

    Returns:
        Message envelope dict
    """
    from remote_protocol_rl import MessageEnvelope

    payload = {
        "checkpoint_id": checkpoint_id,
        "success": success,
        "saved_path": saved_path or "",
    }

    msg = sequencer.create_message(MSG_CHECKPOINT_ACK, payload)
    return msg.to_dict()


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file.

    Args:
        file_path: Path to file

    Returns:
        Hex-encoded SHA256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_chunk_hash(data: bytes) -> str:
    """Compute SHA256 hash of chunk data.

    Args:
        data: Chunk bytes

    Returns:
        Hex-encoded SHA256 hash
    """
    return hashlib.sha256(data).hexdigest()


def chunk_file(file_path: Path, checkpoint_id: str) -> tuple[CheckpointMetadata, List[CheckpointChunk]]:
    """Split file into chunks for transfer.

    Args:
        file_path: Path to checkpoint file
        checkpoint_id: Unique ID for this transfer

    Returns:
        Tuple of (metadata, list of chunks)
    """
    file_size = file_path.stat().st_size
    num_chunks = (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE  # Ceiling division
    file_hash = compute_file_hash(file_path)

    metadata = CheckpointMetadata(
        checkpoint_id=checkpoint_id,
        filename=file_path.name,
        total_size=file_size,
        num_chunks=num_chunks,
        file_hash=file_hash,
        iteration=0,  # Set by caller
        timestamp=time.time(),
    )

    chunks = []
    with open(file_path, 'rb') as f:
        for chunk_index in range(num_chunks):
            data = f.read(CHUNK_SIZE)
            chunk_hash = compute_chunk_hash(data)

            chunk = CheckpointChunk(
                checkpoint_id=checkpoint_id,
                chunk_index=chunk_index,
                total_chunks=num_chunks,
                data=data,
                chunk_hash=chunk_hash,
            )
            chunks.append(chunk)

    return metadata, chunks


def reassemble_chunks(
    chunks: Dict[int, CheckpointChunk],
    metadata: CheckpointMetadata,
    output_path: Path,
) -> bool:
    """Reassemble chunks into complete file and verify.

    Args:
        chunks: Dict mapping chunk_index -> chunk
        metadata: Original checkpoint metadata
        output_path: Where to save reassembled file

    Returns:
        True if reassembly successful and hash matches
    """
    # Verify we have all chunks
    if len(chunks) != metadata.num_chunks:
        print(f"[checkpoint_transfer] Missing chunks: have {len(chunks)}, need {metadata.num_chunks}")
        return False

    # Verify chunk indices are complete
    expected_indices = set(range(metadata.num_chunks))
    actual_indices = set(chunks.keys())
    if expected_indices != actual_indices:
        missing = expected_indices - actual_indices
        print(f"[checkpoint_transfer] Missing chunk indices: {missing}")
        return False

    # Reassemble file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        for chunk_index in range(metadata.num_chunks):
            chunk = chunks[chunk_index]

            # Verify chunk hash
            actual_hash = compute_chunk_hash(chunk.data)
            if actual_hash != chunk.chunk_hash:
                print(f"[checkpoint_transfer] Chunk {chunk_index} hash mismatch!")
                return False

            f.write(chunk.data)

    # Verify complete file hash
    actual_file_hash = compute_file_hash(output_path)
    if actual_file_hash != metadata.file_hash:
        print(f"[checkpoint_transfer] File hash mismatch!")
        print(f"  Expected: {metadata.file_hash}")
        print(f"  Actual:   {actual_file_hash}")
        output_path.unlink()  # Delete corrupted file
        return False

    print(f"[checkpoint_transfer] Successfully reassembled {output_path}")
    print(f"  Size: {output_path.stat().st_size} bytes")
    print(f"  Hash: {actual_file_hash}")
    return True


class CheckpointSender:
    """Sends checkpoint files via NKN with chunking and retry logic."""

    def __init__(self, nkn_bridge, sequencer):
        """Initialize checkpoint sender.

        Args:
            nkn_bridge: NKNSidecar instance
            sequencer: MessageSequencer for ordering
        """
        self.nkn_bridge = nkn_bridge
        self.sequencer = sequencer
        self.active_transfers: Dict[str, tuple[CheckpointMetadata, List[CheckpointChunk]]] = {}

    def send_checkpoint(
        self,
        file_path: Path,
        destination: str,
        iteration: int,
    ) -> str:
        """Initiate checkpoint transfer.

        Args:
            file_path: Path to checkpoint file
            destination: NKN address of receiver
            iteration: Training iteration number

        Returns:
            Checkpoint ID for tracking
        """
        # Generate unique checkpoint ID
        checkpoint_id = f"ckpt_{iteration}_{int(time.time() * 1000)}"

        print(f"[checkpoint_sender] Starting transfer of {file_path.name}")
        print(f"  Checkpoint ID: {checkpoint_id}")
        print(f"  Destination: {destination}")

        # Chunk the file
        metadata, chunks = chunk_file(file_path, checkpoint_id)
        metadata.iteration = iteration

        print(f"  File size: {metadata.total_size} bytes")
        print(f"  Chunks: {metadata.num_chunks}")
        print(f"  Hash: {metadata.file_hash}")

        # Store for potential retransmission
        self.active_transfers[checkpoint_id] = (metadata, chunks)

        # Send metadata
        start_msg = create_checkpoint_start_message(self.sequencer, metadata)
        self.nkn_bridge.send_dm(destination, start_msg)
        print(f"[checkpoint_sender] Sent metadata")

        # Send all chunks
        for chunk in chunks:
            chunk_msg = create_checkpoint_chunk_message(self.sequencer, chunk)
            self.nkn_bridge.send_dm(destination, chunk_msg)
            if chunk.chunk_index % 10 == 0:
                print(f"[checkpoint_sender] Sent chunk {chunk.chunk_index + 1}/{metadata.num_chunks}")

        print(f"[checkpoint_sender] All chunks sent, waiting for confirmation...")
        return checkpoint_id

    def resend_chunks(
        self,
        checkpoint_id: str,
        destination: str,
        chunk_indices: List[int],
    ):
        """Resend specific chunks that were missing.

        Args:
            checkpoint_id: ID of checkpoint transfer
            destination: NKN address of receiver
            chunk_indices: List of chunk indices to resend
        """
        if checkpoint_id not in self.active_transfers:
            print(f"[checkpoint_sender] Unknown checkpoint ID: {checkpoint_id}")
            return

        metadata, chunks = self.active_transfers[checkpoint_id]
        print(f"[checkpoint_sender] Resending {len(chunk_indices)} chunks for {checkpoint_id}")

        for chunk_index in chunk_indices:
            if chunk_index >= len(chunks):
                print(f"[checkpoint_sender] Invalid chunk index: {chunk_index}")
                continue

            chunk = chunks[chunk_index]
            chunk_msg = create_checkpoint_chunk_message(self.sequencer, chunk)
            self.nkn_bridge.send_dm(destination, chunk_msg)

        print(f"[checkpoint_sender] Resent {len(chunk_indices)} chunks")

    def cleanup_transfer(self, checkpoint_id: str):
        """Clean up after successful transfer.

        Args:
            checkpoint_id: ID of completed transfer
        """
        if checkpoint_id in self.active_transfers:
            del self.active_transfers[checkpoint_id]
            print(f"[checkpoint_sender] Cleaned up transfer {checkpoint_id}")


class CheckpointReceiver:
    """Receives checkpoint files via NKN with verification and retry logic."""

    def __init__(self, nkn_bridge, sequencer, save_dir: Path):
        """Initialize checkpoint receiver.

        Args:
            nkn_bridge: NKNSidecar instance
            sequencer: MessageSequencer for ordering
            save_dir: Directory to save received checkpoints
        """
        self.nkn_bridge = nkn_bridge
        self.sequencer = sequencer
        self.save_dir = save_dir
        self.active_transfers: Dict[str, tuple[CheckpointMetadata, Dict[int, CheckpointChunk]]] = {}

    def handle_checkpoint_start(self, sender: str, metadata_dict: Dict[str, Any]):
        """Handle checkpoint transfer initiation.

        Args:
            sender: NKN address of sender
            metadata_dict: Checkpoint metadata
        """
        metadata = CheckpointMetadata(
            checkpoint_id=metadata_dict["checkpoint_id"],
            filename=metadata_dict["filename"],
            total_size=metadata_dict["total_size"],
            num_chunks=metadata_dict["num_chunks"],
            file_hash=metadata_dict["file_hash"],
            iteration=metadata_dict["iteration"],
            timestamp=metadata_dict["timestamp"],
        )

        print(f"[checkpoint_receiver] Starting transfer: {metadata.filename}")
        print(f"  Checkpoint ID: {metadata.checkpoint_id}")
        print(f"  Size: {metadata.total_size} bytes")
        print(f"  Chunks: {metadata.num_chunks}")
        print(f"  Iteration: {metadata.iteration}")

        self.active_transfers[metadata.checkpoint_id] = (metadata, {})

    def handle_checkpoint_chunk(self, sender: str, chunk_dict: Dict[str, Any]):
        """Handle received checkpoint chunk.

        Args:
            sender: NKN address of sender
            chunk_dict: Chunk data
        """
        checkpoint_id = chunk_dict["checkpoint_id"]
        if checkpoint_id not in self.active_transfers:
            print(f"[checkpoint_receiver] Received chunk for unknown transfer: {checkpoint_id}")
            return

        chunk = CheckpointChunk(
            checkpoint_id=checkpoint_id,
            chunk_index=chunk_dict["chunk_index"],
            total_chunks=chunk_dict["total_chunks"],
            data=bytes.fromhex(chunk_dict["data"]),
            chunk_hash=chunk_dict["chunk_hash"],
        )

        metadata, chunks = self.active_transfers[checkpoint_id]
        chunks[chunk.chunk_index] = chunk

        if len(chunks) % 10 == 0 or len(chunks) == metadata.num_chunks:
            print(f"[checkpoint_receiver] Received {len(chunks)}/{metadata.num_chunks} chunks")

        # Check if all chunks received
        if len(chunks) == metadata.num_chunks:
            self._verify_and_save(sender, checkpoint_id, metadata, chunks)

    def _verify_and_save(
        self,
        sender: str,
        checkpoint_id: str,
        metadata: CheckpointMetadata,
        chunks: Dict[int, CheckpointChunk],
    ):
        """Verify chunks and save checkpoint.

        Args:
            sender: NKN address of sender
            checkpoint_id: Checkpoint ID
            metadata: Checkpoint metadata
            chunks: Received chunks
        """
        print(f"[checkpoint_receiver] All chunks received, verifying...")

        # Check for missing chunks
        expected_indices = set(range(metadata.num_chunks))
        actual_indices = set(chunks.keys())
        missing = expected_indices - actual_indices

        if missing:
            print(f"[checkpoint_receiver] Missing chunks: {sorted(missing)}")
            # Request retransmission
            retry_msg = create_checkpoint_request_retry_message(
                self.sequencer,
                checkpoint_id,
                sorted(missing),
            )
            self.nkn_bridge.send_dm(sender, retry_msg)
            return

        # Send completion signal
        complete_msg = create_checkpoint_complete_message(self.sequencer, checkpoint_id)
        self.nkn_bridge.send_dm(sender, complete_msg)
        print(f"[checkpoint_receiver] Sent completion signal")

        # Reassemble and verify
        output_path = self.save_dir / f"iteration_{metadata.iteration}" / metadata.filename
        success = reassemble_chunks(chunks, metadata, output_path)

        # Send ACK
        ack_msg = create_checkpoint_ack_message(
            self.sequencer,
            checkpoint_id,
            success,
            str(output_path) if success else None,
        )
        self.nkn_bridge.send_dm(sender, ack_msg)

        if success:
            print(f"[checkpoint_receiver] ✅ Checkpoint saved: {output_path}")
        else:
            print(f"[checkpoint_receiver] ❌ Checkpoint verification failed!")

        # Cleanup
        del self.active_transfers[checkpoint_id]
