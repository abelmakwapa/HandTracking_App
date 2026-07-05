"""
event_bus.py
============

Minimal synchronous pub/sub bus that decouples the core pipeline from the
optional layers (actions, history, broadcasting, logging).

The pipeline *publishes* facts ("gesture.start", "button.press", ...) and has
no idea who is listening; each optional feature *subscribes* to the topics it
cares about. Disabling a feature = simply not constructing/subscribing it —
the pipeline code is untouched. This is the decoupling seam requested for
the interaction/output layers.

Topics currently published by main.py's pipeline:

    gesture.start   {hand, name, score, t}
    gesture.end     {hand, name, t, duration}
    dynamic         {hand, name, detail, confidence, t}
    interaction     {name, detail, value, t}      (edge-deduped, not per-frame)
    button.press    {name, t}
    landmarks       {t, hands: [...]}             (only with --send-landmarks)

Handlers run synchronously in the video loop, so they must be cheap; anything
slow (network sends, shell commands) must hand off to its own thread/queue —
see broadcaster.py and gesture_actions.py for that pattern.
"""

from collections import defaultdict
from typing import Callable, Dict, List


class EventBus:
    """Tiny topic -> handlers registry. Handlers take one dict payload."""

    def __init__(self):
        self._subs: Dict[str, List[Callable[[dict], None]]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Callable[[dict], None]) -> None:
        self._subs[topic].append(handler)

    def publish(self, topic: str, payload: dict) -> None:
        """Deliver payload to every subscriber of ``topic``.

        A failing subscriber is reported and skipped rather than allowed to
        take down the video loop.
        """
        for handler in self._subs.get(topic, ()):
            try:
                handler(payload)
            except Exception as exc:
                print(f"[warn] event handler for '{topic}' failed: {exc}")
