"""
interaction_analyzer.py
=======================

Two-hand interaction detection.

Unlike the (stateless) gesture classifier, interactions are inherently
*temporal* — a pinch/zoom is only meaningful as a *change* in inter-hand
distance over time, and "which hand is active" depends on recent motion. So
this class keeps a small amount of history between frames.

Interaction rules are written as **pluggable functions** (see ``RULES``): each
takes an ``InteractionContext`` and returns an ``InteractionEvent`` (or None).
To add a new combo (e.g. a two-hand "frame"/rectangle gesture) you just write
another rule function and register it — no changes to the loop below.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional

from geometry import FINGERTIPS, distance, to_xy
from hand_tracker import HandResult


@dataclass
class InteractionContext:
    """Everything a rule needs about the current + recent two-hand state."""
    left: HandResult
    right: HandResult
    palm_distance: float          # current distance between palm centers
    prev_palm_distance: float     # one frame ago (for deltas)
    palm_distance_history: Deque[float]
    left_motion: float            # palm-center displacement since last frame
    right_motion: float


@dataclass
class InteractionEvent:
    name: str
    detail: str = ""
    value: float = 0.0


# ---------------------------------------------------------------------------
# Rule functions. Each is pure w.r.t. the context it is handed.
# ---------------------------------------------------------------------------

TOUCH_DISTANCE = 0.12      # normalized-frame distance for "hands touching"
PINCH_DELTA = 0.015        # per-frame change in palm distance to call it zoom


def rule_touch(ctx: InteractionContext) -> Optional[InteractionEvent]:
    """Hands touching / clapping.

    Trigger if either the palm centers are close OR any fingertip of one hand
    is very near any fingertip of the other (catches high-fives / claps where
    palms face but centers are offset).
    """
    if ctx.palm_distance < TOUCH_DISTANCE:
        return InteractionEvent("touch", "palms close", ctx.palm_distance)

    min_tip = min(
        distance(to_xy(ctx.left.landmarks[i]), to_xy(ctx.right.landmarks[j]))
        for i in FINGERTIPS
        for j in FINGERTIPS
    )
    if min_tip < TOUCH_DISTANCE * 0.8:
        return InteractionEvent("touch", "fingertips close", min_tip)
    return None


def rule_pinch_zoom(ctx: InteractionContext) -> Optional[InteractionEvent]:
    """Two-hand pinch / zoom, detected from the *delta* in inter-hand distance.

    Positive delta (hands moving apart) => zoom-in; negative => zoom-out. We
    compare against a smoothed baseline (front of the history window) so a
    single jittery frame does not fire the event.
    """
    if len(ctx.palm_distance_history) < ctx.palm_distance_history.maxlen:
        return None  # not enough history yet
    baseline = ctx.palm_distance_history[0]
    delta = ctx.palm_distance - baseline
    if abs(delta) < PINCH_DELTA:
        return None
    direction = "zoom_in" if delta > 0 else "zoom_out"
    return InteractionEvent("pinch_zoom", direction, delta)


def rule_active_hand(ctx: InteractionContext) -> Optional[InteractionEvent]:
    """Report which hand shows more recent motion (the 'active' hand)."""
    if max(ctx.left_motion, ctx.right_motion) < 0.005:
        return None  # both essentially still
    if ctx.left_motion > ctx.right_motion:
        return InteractionEvent("active_hand", "Left", ctx.left_motion)
    return InteractionEvent("active_hand", "Right", ctx.right_motion)


# Registry — append your own rule functions here to add combos.
RULES: List[Callable[[InteractionContext], Optional[InteractionEvent]]] = [
    rule_touch,
    rule_pinch_zoom,
    rule_active_hand,
]


class InteractionAnalyzer:
    """Stateful driver that feeds frames to the pluggable rule functions."""

    def __init__(self, history: int = 8):
        self._palm_hist: Deque[float] = deque(maxlen=history)
        self._prev_palm_distance: Optional[float] = None
        self._prev_left_center = None
        self._prev_right_center = None

    def reset(self) -> None:
        """Clear temporal state (call when hands leave the frame)."""
        self._palm_hist.clear()
        self._prev_palm_distance = None
        self._prev_left_center = None
        self._prev_right_center = None

    @staticmethod
    def _split_hands(hands: List[HandResult]):
        """Return (left, right) HandResults or (None, None) if not exactly one each."""
        left = next((h for h in hands if h.label == "Left"), None)
        right = next((h for h in hands if h.label == "Right"), None)
        return left, right

    def analyze(self, hands: List[HandResult]) -> List[InteractionEvent]:
        """Run all rules for the current frame; returns list of fired events."""
        left, right = self._split_hands(hands)
        if left is None or right is None:
            self.reset()
            return []

        lc, rc = left.palm_center, right.palm_center
        palm_dist = distance(lc, rc)

        left_motion = (
            distance(lc, self._prev_left_center) if self._prev_left_center else 0.0
        )
        right_motion = (
            distance(rc, self._prev_right_center) if self._prev_right_center else 0.0
        )

        ctx = InteractionContext(
            left=left,
            right=right,
            palm_distance=palm_dist,
            prev_palm_distance=self._prev_palm_distance
            if self._prev_palm_distance is not None
            else palm_dist,
            palm_distance_history=self._palm_hist,
            left_motion=left_motion,
            right_motion=right_motion,
        )

        events = [ev for rule in RULES if (ev := rule(ctx)) is not None]

        # Update temporal state *after* the rules have read the old values.
        self._palm_hist.append(palm_dist)
        self._prev_palm_distance = palm_dist
        self._prev_left_center = lc
        self._prev_right_center = rc
        return events
