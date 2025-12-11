from __future__ import annotations

import base64
import contextlib
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
BRIDGE_DIR = PROJECT_ROOT / "service_router" / "bridge-node"
BRIDGE_JS = BRIDGE_DIR / "nkn_bridge.js"
PKG_JSON = BRIDGE_DIR / "package.json"
NODE_MODULE = BRIDGE_DIR / "node_modules" / "nkn-sdk"

BRIDGE_SRC = r"""'use strict';
const nkn = require('nkn-sdk');
const readline = require('readline');

const SEED_HEX = (process.env.NKN_SEED_HEX || '').toLowerCase().replace(/^0x/,'');
const IDENT = String(process.env.NKN_IDENTIFIER || 'relay');
const NUM = parseInt(process.env.NKN_NUM_SUBCLIENTS || '2', 10) || 2;
const SEED_WS = (process.env.NKN_BRIDGE_SEED_WS || '').split(',').map(s=>s.trim()).filter(Boolean);
const PROBE_EVERY_MS = parseInt(process.env.NKN_SELF_PROBE_MS || '12000', 10);
const PROBE_FAILS_EXIT = parseInt(process.env.NKN_SELF_PROBE_FAILS || '3', 10);

function out(obj){
  try{ process.stdout.write(JSON.stringify(obj)+'\n'); }
  catch(e){ /* ignore */ }
}

function spawn(){
  if(!/^[0-9a-f]{64}$/.test(SEED_HEX)){
    out({type:'crit', msg:'bad seed'});
    process.exit(1);
  }

  const client = new nkn.MultiClient({
    seed: SEED_HEX,
    identifier: IDENT,
    numSubClients: NUM,
    seedWsAddr: SEED_WS.length ? SEED_WS : undefined,
    wsConnHeartbeatTimeout: 120000,
  });

  let probeFails = 0;
  let probeTimer = null;
  function startProbe(){
    stopProbe();
    probeTimer = setInterval(async ()=>{
      try {
        await client.send(String(client.addr||''), JSON.stringify({event:'relay.selfprobe', ts: Date.now()}), {noReply:true});
        probeFails = 0;
        out({type:'status', state:'probe_ok'});
      } catch (e){
        probeFails++;
        out({type:'status', state:'probe_fail', fails:probeFails, msg:String(e&&e.message||e)});
        if (probeFails >= PROBE_FAILS_EXIT){
          out({type:'status', state:'probe_exit'});
          process.exit(3);
        }
      }
    }, PROBE_EVERY_MS);
  }
  function stopProbe(){
    if (probeTimer){ clearInterval(probeTimer); probeTimer=null; }
  }

  client.on('connect', ()=>{
    out({type:'ready', address:String(client.addr||''), ts: Date.now()});
    startProbe();
  });
  client.on('error', (e)=>{ out({type:'status', state:'error', msg:String(e&&e.message||e)}); process.exit(2); });
  client.on('close', ()=>{ out({type:'status', state:'close'}); process.exit(2); });

  client.on('message', (a, b)=>{
    try{
      let src, payload;
      if (a && typeof a==='object' && a.payload!==undefined){ src=String(a.src||''); payload=a.payload; }
      else { src=String(a||''); payload=b; }
      const s = Buffer.isBuffer(payload) ? payload.toString('utf8') : (typeof payload==='string'? payload : String(payload));
      let parsed=null; try{ parsed=JSON.parse(s); }catch{}
      out({type:'nkn-dm', src, msg: parsed || {event:'<non-json>', raw:s}});
    }catch(e){ out({type:'err', msg:String(e&&e.message||e)}); }
  });

  const rl = readline.createInterface({input: process.stdin});
  rl.on('line', line=>{
    let cmd; try{ cmd=JSON.parse(line); }catch{return; }
    if(cmd && cmd.type==='dm' && cmd.to && cmd.data){
      const opts = cmd.opts || {noReply:true};
      client.send(cmd.to, JSON.stringify(cmd.data), opts).catch(err=>{
        out({type:'status', state:'send_error', msg:String(err&&err.message||err)});
      });
    }
  });

  process.on('exit', ()=>{ stopProbe(); if(probeTimer){ clearInterval(probeTimer);} });
  process.on('unhandledRejection', e=>{ out({type:'status', state:'unhandledRejection', msg:String(e)}); process.exit(1); });
  process.on('uncaughtException', e=>{ out({type:'status', state:'uncaughtException', msg:String(e)}); process.exit(1); });
}

spawn();
"""

SEND_QUEUE_MAX = 2048

def ensure_nkn_bridge() -> None:
    if not shutil.which("node"):
        raise RuntimeError("Node.js binary 'node' not found; install Node.js to use NKN transports.")
    if not shutil.which("npm"):
        raise RuntimeError("npm binary 'npm' not found; install npm to use NKN transports.")
    BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    if not PKG_JSON.exists():
        subprocess.check_call(["npm", "init", "-y"], cwd=BRIDGE_DIR)
    if not BRIDGE_JS.exists() or BRIDGE_JS.read_text(encoding="utf-8") != BRIDGE_SRC:
        BRIDGE_JS.write_text(BRIDGE_SRC, encoding="utf-8")
    if not NODE_MODULE.exists():
        subprocess.check_call(["npm", "install", "nkn-sdk@^1.3.6"], cwd=BRIDGE_DIR)


class NKNSidecar:
    def __init__(
        self,
        *,
        seed_hex: str,
        identifier: str,
        num_subclients: int = 2,
        seed_ws: str = "",
        self_probe_ms: int = 12000,
        self_probe_fails: int = 3,
        on_message: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_ready: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.seed_hex = seed_hex.lower().replace("0x", "")
        self.identifier = identifier
        self.num_subclients = max(1, num_subclients)
        self.seed_ws = seed_ws
        self.self_probe_ms = self_probe_ms
        self.self_probe_fails = self_probe_fails
        self.on_message = on_message
        self.on_ready = on_ready
        self.on_status = on_status
        self.on_error = on_error
        self.proc: Optional[subprocess.Popen[str]] = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.send_queue: "queue.Queue[tuple[str, Dict[str, Any], Dict[str, Any]]]" = queue.Queue(maxsize=SEND_QUEUE_MAX)
        self.ready_event = threading.Event()
        self.address = ""
        self.state = ""
        self._bytes_in = 0
        self._bytes_out = 0
        self._messages_in = 0
        self._messages_out = 0
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._sender_thread: Optional[threading.Thread] = None
        self._chunk_limit = int(os.environ.get("NKN_DM_MAX_BYTES", "900000"))
        self._chunk_buffers: Dict[str, Dict[str, Any]] = {}
        self._chunk_lock = threading.Lock()

    def start(self) -> None:
        ensure_nkn_bridge()
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return
            env = os.environ.copy()
            env["NKN_SEED_HEX"] = self.seed_hex
            env["NKN_IDENTIFIER"] = self.identifier
            env["NKN_NUM_SUBCLIENTS"] = str(self.num_subclients)
            if self.seed_ws:
                env["NKN_BRIDGE_SEED_WS"] = self.seed_ws
            env["NKN_SELF_PROBE_MS"] = str(self.self_probe_ms)
            env["NKN_SELF_PROBE_FAILS"] = str(self.self_probe_fails)
            self.stop_event.clear()
            self.ready_event.clear()
            self.address = ""
            self.state = "starting"
            self.proc = subprocess.Popen(
                ["node", str(BRIDGE_JS)],
                cwd=BRIDGE_DIR,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        self._stdout_thread = threading.Thread(target=self._stdout_pump, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_pump, daemon=True)
        self._stderr_thread.start()
        if not self._sender_thread:
            self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
            self._sender_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self.lock:
            proc = self.proc
        if proc:
            with contextlib.suppress(Exception):
                if proc.stdin:
                    proc.stdin.close()
            with contextlib.suppress(Exception):
                proc.terminate()
        if self._stdout_thread and self._stdout_thread.is_alive():
            self._stdout_thread.join(timeout=1.0)
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1.0)

    def wait_ready(self, timeout: float = 30.0) -> bool:
        return self.ready_event.wait(timeout)

    def send_dm(self, to: str, data: Dict[str, Any], opts: Optional[Dict[str, Any]] = None) -> None:
        opts = opts or {}
        try:
            raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
        except Exception:
            raw = b""

        if raw and len(raw) > self._chunk_limit:
            chunk_id = uuid.uuid4().hex
            chunk_size = max(1, self._chunk_limit - 4096)
            total = (len(raw) + chunk_size - 1) // chunk_size
            for idx in range(total):
                start = idx * chunk_size
                end = min(len(raw), start + chunk_size)
                part = raw[start:end]
                chunk_msg = {
                    "__chunk": True,
                    "chunk_id": chunk_id,
                    "idx": idx,
                    "total": total,
                    "data": base64.b64encode(part).decode("ascii"),
                }
                self._enqueue_payload((to, chunk_msg, opts))
        else:
            self._enqueue_payload((to, data, opts))

    def _enqueue_payload(self, payload: tuple[str, Dict[str, Any], Dict[str, Any]]) -> None:
        to, data, opts = payload
        try:
            self.send_queue.put_nowait(payload)
        except queue.Full:
            with contextlib.suppress(queue.Full):
                _ = self.send_queue.get_nowait()
            self.send_queue.put(payload)
        self._messages_out += 1
        try:
            self._bytes_out += len(json.dumps({"to": to, "data": data, "opts": opts}).encode("utf-8"))
        except Exception:
            pass

    def bytes_in(self) -> int:
        return self._bytes_in

    def bytes_out(self) -> int:
        return self._bytes_out

    def messages_in(self) -> int:
        return self._messages_in

    def messages_out(self) -> int:
        return self._messages_out

    def _stdout_pump(self) -> None:
        proc = self.proc
        if not proc or not proc.stdout:
            return
        while not self.stop_event.is_set():
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.01)
                continue
            self._messages_in += 1
            self._bytes_in += len(line.encode("utf-8"))
            try:
                msg = json.loads(line.strip())
            except Exception:
                continue
            typ = msg.get("type")
            if typ == "ready":
                self.address = msg.get("address", "")
                self.state = "ready"
                self.ready_event.set()
                if self.on_ready:
                    self.on_ready(self.address)
            elif typ == "status":
                state = msg.get("state", "")
                if state:
                    self.state = state
                if self.on_status:
                    self.on_status(f"bridge {state}")
            elif typ == "nkn-dm":
                src = msg.get("src", "")
                body = msg.get("msg", {})
                if self.on_message and isinstance(body, dict):
                    reconstructed = self._handle_chunk(src, body)
                    if reconstructed is not None:
                        self.on_message(src, reconstructed)
            elif typ == "err":
                if self.on_error:
                    self.on_error(msg.get("msg", "bridge error"))
        self.ready_event.clear()
        self.state = "stopped"

    def _stderr_pump(self) -> None:
        proc = self.proc
        if not proc or not proc.stderr:
            return
        while not self.stop_event.is_set():
            line = proc.stderr.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.01)
                continue
            if self.on_error:
                self.on_error(line.strip())

    def _sender_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                to, data, opts = self.send_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            wrote = False
            while not wrote and not self.stop_event.is_set():
                with self.lock:
                    proc = self.proc
                    stdin = proc.stdin if proc else None
                if proc and proc.poll() is None and stdin:
                    try:
                        payload = {"type": "dm", "to": to, "data": data}
                        if opts:
                            payload["opts"] = opts
                        stdin.write(json.dumps(payload) + "\n")
                        stdin.flush()
                        wrote = True
                    except Exception:
                        time.sleep(0.1)
                else:
                    time.sleep(0.2)
            self.send_queue.task_done()

    def _handle_chunk(self, src: str, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Reassemble chunked DMs produced by send_dm when payload exceeds limit."""
        if "__chunk" not in body:
            return body
        try:
            chunk_id = str(body.get("chunk_id"))
            total = int(body.get("total", 0))
            idx = int(body.get("idx", -1))
            data_b64 = body.get("data", "")
        except Exception:
            return None
        if not chunk_id or total <= 0 or idx < 0 or idx >= total:
            return None
        try:
            part = base64.b64decode(data_b64.encode("ascii"))
        except Exception:
            return None
        with self._chunk_lock:
            buf = self._chunk_buffers.get(chunk_id)
            if buf is None:
                buf = {"total": total, "parts": {}, "ts": time.time()}
                self._chunk_buffers[chunk_id] = buf
            buf["parts"][idx] = part
            # cleanup stale buffers older than 5 minutes
            cutoff = time.time() - 300
            self._chunk_buffers = {
                cid: b for cid, b in self._chunk_buffers.items() if b.get("ts", 0) >= cutoff
            }
            if len(buf["parts"]) < total:
                return None
            ordered = [buf["parts"][i] for i in range(total)]
            raw = b"".join(ordered)
            self._chunk_buffers.pop(chunk_id, None)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None
