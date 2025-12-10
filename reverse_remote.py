"""Reverse connection bridge that accepts remote agents initiating connections."""

import socketserver
import threading
import uuid
from typing import Callable, Iterable, List, Optional, Tuple

import remote_protocol

LogCallback = Callable[[str], None]
AckCallback = Callable[[str], None]


class ReverseHandler(socketserver.StreamRequestHandler):
    def __init__(self, *args, bridge: "ReverseBridge", **kwargs):
        self.bridge = bridge
        self.reader = None
        self.writer = None
        self._exit_event = threading.Event()
        self.last_exit_code: Optional[int] = None
        super().__init__(*args, **kwargs)

    def handle(self) -> None:
        self.reader = self.rfile
        self.writer = self.wfile
        self.bridge.register_handler(self)
        try:
            for msg in remote_protocol.iter_messages(self.reader):
                typ = msg.get("type")
                if typ == "log":
                    self.bridge.log_callback(msg.get("message", ""))
                elif typ == "start":
                    desc = msg.get("description", "run started")
                    self.bridge.log_callback(f"[remote] {desc}")
                elif typ == "ack":
                    self.bridge.ack_callback(msg.get("message", "ack"))
                elif typ == "exit":
                    self.last_exit_code = int(msg.get("code", 0))
                    self._exit_event.set()
                elif typ == "handshake":
                    self.bridge.log_callback(f"Remote handshake: {msg.get('description', '')}")
        finally:
            self.bridge.unregister_handler(self)

    def send_command(self, cmd: Iterable[str], description: Optional[str]) -> int:
        if not self.writer:
            raise RuntimeError("Reverse handler writer not ready.")
        payload = {
            "type": "command",
            "session_id": str(uuid.uuid4()),
            "description": description or "reverse run",
            "cmd": list(cmd),
        }
        self._exit_event.clear()
        self.last_exit_code = None
        self.writer.write(remote_protocol.encode_message(payload))
        self.writer.flush()
        if not self._exit_event.wait(self.bridge.command_timeout):
            raise RuntimeError("Remote command timed out.")
        return self.last_exit_code if self.last_exit_code is not None else 1


class ReverseServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


class ReverseBridge:
    def __init__(
        self,
        host: str,
        port: int,
        log_callback: LogCallback,
        ack_callback: AckCallback,
        command_timeout: float = 300.0,
    ):
        self.host = host
        self.port = port
        self.log_callback = log_callback
        self.ack_callback = ack_callback
        self.command_timeout = command_timeout
        self._server: Optional[ReverseServer] = None
        self._thread: Optional[threading.Thread] = None
        self._handler: Optional[ReverseHandler] = None
        self._handler_lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        handler_factory = lambda *args, **kwargs: ReverseHandler(*args, bridge=self, **kwargs)
        self._server = ReverseServer((self.host, self.port), handler_factory)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._running = True

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._running = False

    def register_handler(self, handler: ReverseHandler) -> None:
        with self._handler_lock:
            self._handler = handler
        self.ack_callback(f"Remote agent connected from {handler.client_address}")
        self.log_callback("Waiting for commands...")

    def unregister_handler(self, handler: ReverseHandler) -> None:
        with self._handler_lock:
            if self._handler is handler:
                self._handler = None

    def has_handler(self) -> bool:
        with self._handler_lock:
            return self._handler is not None

    def should_continue(self) -> bool:
        return self._running

    def dispatch_command(self, cmd: Iterable[str], description: Optional[str]) -> int:
        with self._handler_lock:
            if not self._handler:
                raise RuntimeError("No remote agent connected.")
            handler = self._handler
        return handler.send_command(cmd, description)


_bridge: Optional[ReverseBridge] = None


def ensure_bridge(host: str, port: int, log_callback: LogCallback, ack_callback: AckCallback) -> ReverseBridge:
    global _bridge
    if _bridge is None:
        _bridge = ReverseBridge(
            host=host,
            port=port,
            log_callback=log_callback,
            ack_callback=ack_callback,
        )
        _bridge.start()
    return _bridge


def dispatch_command(cmd: Iterable[str], description: Optional[str]) -> int:
    if _bridge is None:
        raise RuntimeError("Reverse bridge is not running.")
    return _bridge.dispatch_command(cmd, description)


def is_agent_available() -> bool:
    return _bridge is not None and _bridge.has_handler()


def bridge_address() -> Tuple[str, int]:
    if _bridge is None:
        raise RuntimeError("Reverse bridge is not running.")
    return (_bridge.host, _bridge.port)
