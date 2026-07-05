"""
broadcaster.py
==============

Real-time event output to external apps (Processing, TouchDesigner, browser)
over OSC (UDP) and/or WebSocket.

Non-blocking by construction: bus events are pushed onto a bounded queue and
a single daemon worker thread does all network I/O. The video loop never
touches a socket, so a slow/absent consumer cannot drop the frame rate — if
the queue backs up, we drop *events* (with a periodic warning), never frames.

Wire formats
------------
* **OSC** (pure stdlib, UDP): address ``/handtrack/<topic with . -> />``,
  type tag ``,s``, single string argument containing the JSON payload.
  Example: topic ``gesture.start`` -> address ``/handtrack/gesture/start``,
  arg ``{"hand": "Right", "name": "fist", "score": 0.91, "t": 12.34}``.
  A JSON-in-string arg keeps the schema uniform across topics; both
  TouchDesigner and Processing's oscP5 can parse JSON from a string arg.
* **WebSocket** (requires the optional ``websockets`` package): each event is
  one text frame of JSON: ``{"topic": "...", ...payload}``. Connect any
  number of clients to ``ws://<host>:<port>/``.

The ``landmarks`` topic is high-rate (every frame); it is only published when
main.py is run with --send-landmarks, so plain gesture/interaction use stays
lightweight.
"""

import json
import queue
import socket
import threading
from functools import partial

TOPICS = ("gesture.start", "gesture.end", "dynamic", "interaction", "button.press", "landmarks")


def _osc_encode(address: str, payload_json: str) -> bytes:
    """Encode one OSC message with a single string argument.

    OSC strings are null-terminated and zero-padded to a 4-byte boundary.
    """
    def pad(b: bytes) -> bytes:
        b += b"\x00"
        return b + b"\x00" * ((4 - len(b) % 4) % 4)

    return pad(address.encode()) + pad(b",s") + pad(payload_json.encode())


class _WsServer:
    """Tiny broadcast-only WebSocket server on its own asyncio loop/thread."""

    def __init__(self, port: int):
        try:
            import asyncio
            import websockets
        except ImportError:
            raise RuntimeError(
                "the 'websockets' package is required for --ws-port "
                "(pip install websockets)"
            )
        self._asyncio = asyncio
        self._clients = set()
        self._loop = asyncio.new_event_loop()

        async def handler(ws):
            self._clients.add(ws)
            try:
                await ws.wait_closed()
            finally:
                self._clients.discard(ws)

        def run():
            asyncio.set_event_loop(self._loop)

            # Newer websockets releases (>=14) require serve() to be created
            # while the event loop is running, so start it from a coroutine.
            async def start():
                await websockets.serve(handler, "0.0.0.0", port)

            self._loop.run_until_complete(start())
            self._loop.run_forever()

        threading.Thread(target=run, daemon=True, name="ws-server").start()
        print(f"[info] WebSocket broadcast on ws://localhost:{port}/")

    def send(self, text: str) -> None:
        if not self._clients:
            return

        async def _broadcast():
            for ws in list(self._clients):
                try:
                    await ws.send(text)
                except Exception:
                    self._clients.discard(ws)

        self._asyncio.run_coroutine_threadsafe(_broadcast(), self._loop)


class Broadcaster:
    """Subscribes to all pipeline topics and relays them off-thread.

    Args:
        bus: the EventBus to subscribe on.
        osc_target: "host:port" for OSC-over-UDP output, or None.
        ws_port: local port for the WebSocket server, or None.
    """

    QUEUE_SIZE = 256

    def __init__(self, bus, osc_target=None, ws_port=None):
        self._q: queue.Queue = queue.Queue(maxsize=self.QUEUE_SIZE)
        self._drops = 0

        self._osc_sock = None
        self._osc_addr = None
        if osc_target:
            host, _, port = osc_target.rpartition(":")
            if not host or not port.isdigit():
                raise SystemExit(
                    f"[error] --osc expects host:port, got {osc_target!r} "
                    f"(e.g. --osc 127.0.0.1:9000)"
                )
            self._osc_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._osc_addr = (host, int(port))
            print(f"[info] OSC broadcast to {host}:{port}")

        self._ws = _WsServer(ws_port) if ws_port else None

        for topic in TOPICS:
            bus.subscribe(topic, partial(self._enqueue, topic))
        threading.Thread(target=self._worker, daemon=True, name="broadcaster").start()

    # -- video-loop side (must never block) ----------------------------------
    def _enqueue(self, topic: str, payload: dict) -> None:
        try:
            self._q.put_nowait((topic, payload))
        except queue.Full:
            self._drops += 1
            if self._drops % 100 == 1:
                print(
                    f"[warn] Broadcast queue full — dropped {self._drops} events "
                    f"(is the consumer keeping up?)."
                )

    # -- worker-thread side ---------------------------------------------------
    def _worker(self) -> None:
        while True:
            topic, payload = self._q.get()
            text = json.dumps({"topic": topic, **payload}, separators=(",", ":"))
            if self._osc_sock is not None:
                address = "/handtrack/" + topic.replace(".", "/")
                try:
                    self._osc_sock.sendto(_osc_encode(address, text), self._osc_addr)
                except OSError as exc:
                    self._drops += 1
                    if self._drops % 100 == 1:
                        print(f"[warn] OSC send failed: {exc}")
            if self._ws is not None:
                self._ws.send(text)
