"""
gesture_history.py
==================

On-screen rolling log of recent gesture/interaction events, for debugging
and live demos ("did that swipe actually register?").

Subscribes to the event bus; the pipeline itself never references this class,
so it can be disabled (--no-history) without touching anything else.

Performance: rendering is a handful of cv2.putText calls per frame (bounded
by ``max_items``), which is negligible next to landmark inference. Event
*ingestion* only happens when events fire (a few per second at most), so no
throttling is needed.
"""

from collections import deque
from typing import Deque, Tuple

import cv2

_COLOR = (200, 200, 200)
_HEADER_COLOR = (0, 255, 255)


class GestureHistory:
    """Rolling event log rendered as a right-side overlay panel."""

    def __init__(self, bus, max_items: int = 8):
        self._items: Deque[Tuple[float, str]] = deque(maxlen=max_items)
        for topic in ("gesture.start", "gesture.end", "dynamic", "interaction", "button.press"):
            bus.subscribe(topic, self._make_handler(topic))

    def _make_handler(self, topic: str):
        def handler(p: dict) -> None:
            t = p.get("t", 0.0)
            if topic == "gesture.start":
                text = f"{p['hand'][0]}: {p['name']} ON"
            elif topic == "gesture.end":
                text = f"{p['hand'][0]}: {p['name']} off ({p['duration']:.1f}s)"
            elif topic == "dynamic":
                text = f"{p['hand'][0]}: {p['name']} {p['detail']}"
            elif topic == "interaction":
                text = f"2H: {p['name']} {p['detail']}"
            else:  # button.press
                text = f"BTN: {p['name']}"
            self._items.append((t, text))

        return handler

    def draw(self, frame) -> None:
        h, w = frame.shape[:2]
        x = w - 330
        y = 90  # below the cursor-status line
        cv2.putText(frame, "-- events --", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, _HEADER_COLOR, 1, cv2.LINE_AA)
        for t, text in reversed(self._items):  # newest at top
            y += 20
            cv2.putText(frame, f"{t:7.2f}s  {text}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLOR, 1, cv2.LINE_AA)
