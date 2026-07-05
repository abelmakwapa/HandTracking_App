"""
dynamic_gestures.py
====================

Motion-based ("dynamic") gesture recognition, layered on top of the
per-frame palm-center positions that HandTracker already computes.

Unlike GestureClassifier (which looks at a single frame's finger geometry),
these gestures only make sense as a *trajectory* over time: a swipe is a
displacement, a wave is an oscillation, a circle is an angular sweep. We keep
a short rolling history of (timestamp, position) per hand and evaluate it
each frame.

Design mirrors interaction_analyzer.py: one method per gesture, each
returning an event (not a bare bool), so new motion gestures can be added
without touching the driver loop. A per-gesture cooldown, plus clearing the
trajectory on a fire, stops one physical motion from spamming repeat events
while it's still being performed.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

Point = Tuple[float, float]

# --- Tunable thresholds (see README "Tuning" section) -----------------------
HISTORY_SECONDS = 1.5        # trajectory history retained per hand
SWIPE_WINDOW = 0.5           # a swipe must complete within this many seconds
SWIPE_MIN_DISTANCE = 0.25    # net displacement (normalized frame units) for a swipe
SWIPE_AXIS_RATIO = 1.8       # dominant-axis displacement must beat the other axis by this factor
WAVE_MIN_REVERSALS = 3       # x-direction changes within the window to call it a wave
WAVE_MIN_AMPLITUDE = 0.04    # min x movement per half-cycle to count as a reversal (filters jitter)
WAVE_MAX_Y_DRIFT = 0.15      # wrist shouldn't travel far vertically during a wave
CIRCLE_MIN_ANGLE = 300.0     # cumulative swept angle (degrees) to call it a full circle
CIRCLE_MIN_RADIUS = 0.05     # min average radius from centroid (filters out hand jitter in place)
GESTURE_COOLDOWN = 0.8       # seconds to suppress re-firing the same gesture on the same hand


@dataclass
class DynamicGestureEvent:
    """A fired motion gesture.

    Attributes:
        name:       "swipe", "wave", or "circle".
        detail:     gesture-specific qualifier, e.g. "left"/"right"/"up"/"down"
                    for swipe, "clockwise"/"counterclockwise" for circle.
        hand_label: "Left" or "Right" — which hand performed it.
        confidence: [0, 1] score for how cleanly the motion matched the
                    pattern (not a hard boolean, so callers can filter weak
                    detections without editing thresholds here).
    """

    name: str
    detail: str
    hand_label: str
    confidence: float = 1.0


@dataclass
class _HandHistory:
    positions: Deque[Tuple[float, Point]] = field(default_factory=deque)
    last_fired: Dict[str, float] = field(default_factory=dict)


class DynamicGestureRecognizer:
    """Tracks per-hand motion trajectories and detects swipes/waves/circles.

    Call ``update(label, palm_center)`` once per visible hand per frame; it
    returns a ``DynamicGestureEvent`` when a motion gesture completes, or
    ``None`` otherwise.
    """

    def __init__(self, history_seconds: float = HISTORY_SECONDS):
        self._history_seconds = history_seconds
        self._hands: Dict[str, _HandHistory] = {}

    def prune_absent(self, present_labels) -> None:
        """Drop trajectory state for any hand no longer in frame.

        Without this, a hand that leaves and re-enters the frame could splice
        its old trajectory onto new positions and fire a false gesture from
        the jump between them.
        """
        for label in list(self._hands):
            if label not in present_labels:
                del self._hands[label]

    def _hist(self, label: str) -> _HandHistory:
        return self._hands.setdefault(label, _HandHistory())

    def _prune_old(self, hist: _HandHistory, now: float) -> None:
        cutoff = now - self._history_seconds
        while hist.positions and hist.positions[0][0] < cutoff:
            hist.positions.popleft()

    def _on_cooldown(self, hist: _HandHistory, name: str, now: float) -> bool:
        last = hist.last_fired.get(name)
        return last is not None and (now - last) < GESTURE_COOLDOWN

    def _fire(self, hist: _HandHistory, now: float, name: str) -> None:
        hist.last_fired[name] = now
        # Consume the trajectory so the same physical motion can't
        # immediately re-trigger a second event on the very next frame.
        hist.positions.clear()

    # -- one method per gesture ----------------------------------------------
    def _check_swipe(
        self, hist: _HandHistory, now: float, label: str
    ) -> Optional[DynamicGestureEvent]:
        """Fast, mostly-one-axis displacement within a short time window."""
        if self._on_cooldown(hist, "swipe", now):
            return None

        window_start = now - SWIPE_WINDOW
        recent = [p for (t, p) in hist.positions if t >= window_start]
        if len(recent) < 2:
            return None

        p0, p1 = recent[0], recent[-1]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        adx, ady = abs(dx), abs(dy)

        if adx >= ady * SWIPE_AXIS_RATIO and adx >= SWIPE_MIN_DISTANCE:
            direction = "right" if dx > 0 else "left"
            magnitude = adx
        elif ady >= adx * SWIPE_AXIS_RATIO and ady >= SWIPE_MIN_DISTANCE:
            # Screen y grows downward.
            direction = "down" if dy > 0 else "up"
            magnitude = ady
        else:
            return None  # too diagonal, or too small, to call confidently

        confidence = min(1.0, magnitude / (SWIPE_MIN_DISTANCE * 2))
        self._fire(hist, now, "swipe")
        return DynamicGestureEvent("swipe", direction, label, confidence)

    def _check_wave(
        self, hist: _HandHistory, now: float, label: str
    ) -> Optional[DynamicGestureEvent]:
        """Repeated left-right x oscillation with limited vertical drift."""
        if self._on_cooldown(hist, "wave", now):
            return None
        pts = list(hist.positions)
        if len(pts) < 6:
            return None

        xs = [p[1][0] for p in pts]
        ys = [p[1][1] for p in pts]
        if max(ys) - min(ys) > WAVE_MAX_Y_DRIFT:
            return None  # too much vertical motion to be a wave

        # Count direction reversals in x, ignoring sub-threshold jitter.
        reversals = 0
        direction = None
        last_extreme = xs[0]
        for x in xs[1:]:
            delta = x - last_extreme
            if abs(delta) < WAVE_MIN_AMPLITUDE:
                continue
            new_direction = "right" if delta > 0 else "left"
            if direction is not None and new_direction != direction:
                reversals += 1
            direction = new_direction
            last_extreme = x

        if reversals < WAVE_MIN_REVERSALS:
            return None

        confidence = min(1.0, reversals / (WAVE_MIN_REVERSALS + 2))
        self._fire(hist, now, "wave")
        return DynamicGestureEvent("wave", f"{reversals} reversals", label, confidence)

    def _check_circle(
        self, hist: _HandHistory, now: float, label: str
    ) -> Optional[DynamicGestureEvent]:
        """Cumulative signed angle swept around the trajectory's own centroid.

        We sum the signed angle between consecutive points as seen from the
        centroid; a full loop accumulates close to +-360 degrees. Angle
        deltas are normalized into (-pi, pi] before summing so a wraparound
        from +179 to -179 degrees is treated as a small step, not a reversal.
        The sign of the total gives the rotation direction.
        """
        if self._on_cooldown(hist, "circle", now):
            return None
        pts = [p[1] for p in hist.positions]
        if len(pts) < 8:
            return None

        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        radii = [math.hypot(p[0] - cx, p[1] - cy) for p in pts]
        if (sum(radii) / len(radii)) < CIRCLE_MIN_RADIUS:
            return None  # motion too small/tight to be a deliberate circle

        angles = [math.atan2(p[1] - cy, p[0] - cx) for p in pts]
        total = 0.0
        for a0, a1 in zip(angles, angles[1:]):
            d = a1 - a0
            while d > math.pi:
                d -= 2 * math.pi
            while d < -math.pi:
                d += 2 * math.pi
            total += d

        swept_degrees = math.degrees(abs(total))
        if swept_degrees < CIRCLE_MIN_ANGLE:
            return None

        confidence = min(1.0, swept_degrees / 360.0)
        direction = "clockwise" if total > 0 else "counterclockwise"
        self._fire(hist, now, "circle")
        return DynamicGestureEvent("circle", direction, label, confidence)

    # -- driver ---------------------------------------------------------------
    def update(
        self, label: str, palm_center: Point, now: Optional[float] = None
    ) -> Optional[DynamicGestureEvent]:
        """Feed one hand's current palm center; returns a fired event, if any.

        Checked in order swipe -> wave -> circle. A firing check clears the
        trajectory (see ``_fire``), so at most one gesture fires per motion.

        Args:
            now: pipeline timestamp in seconds. Pass the recorded timestamp
                during session playback so detection is deterministic;
                defaults to the wall clock for live use.
        """
        if now is None:
            now = time.perf_counter()
        hist = self._hist(label)
        hist.positions.append((now, palm_center))
        self._prune_old(hist, now)

        for check in (self._check_swipe, self._check_wave, self._check_circle):
            event = check(hist, now, label)
            if event is not None:
                return event
        return None
