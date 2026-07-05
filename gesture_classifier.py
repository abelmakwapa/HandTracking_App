"""
gesture_classifier.py
=====================

Static (single-frame) gesture recognition from hand landmarks.

Design notes
------------
Every gesture method returns a **confidence score in [0, 1]** instead of a hard
boolean. That lets ``classify()`` pick the best match *and* lets you later tune
thresholds, blend scores over time, or reject low-confidence frames — none of
which is possible with a bare True/False.

Scores are built from smooth "membership" functions rather than step functions:
we map a geometric quantity (a joint angle, or a normalized distance) through a
soft ramp so that, e.g., a finger that is *almost* straight scores ~0.9 rather
than flipping from 0 to 1 at an arbitrary cutoff.

All distances are normalized by ``hand_scale`` (palm length) so the classifier
behaves identically whether your hand is near or far from the camera.
"""

from dataclasses import dataclass
from typing import Dict

from geometry import (
    FINGER_JOINTS,
    THUMB_CMC,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    WRIST,
    angle,
    distance,
    hand_scale,
    to_xy,
)

# --- Tunable thresholds (see README "Tuning" section) ----------------------
EXTENDED_ANGLE = 160.0   # PIP angle (deg) above which a finger counts as straight
CURLED_ANGLE = 100.0     # PIP angle (deg) below which a finger counts as curled
THUMB_EXT_ANGLE = 150.0  # IP angle (deg) for the thumb being straight
OK_TOUCH_DIST = 0.35     # thumb-index tip distance (in palm-lengths) for "touching"


def _ramp(value: float, low: float, high: float) -> float:
    """Linearly map ``value`` from 0 at ``low`` to 1 at ``high`` (clamped).

    Works in either direction: if ``low > high`` the ramp is descending, which
    is handy for "smaller is better" quantities like a pinch distance.
    """
    if low == high:
        return 1.0 if value >= low else 0.0
    t = (value - low) / (high - low)
    return max(0.0, min(1.0, t))


@dataclass
class Gesture:
    name: str
    score: float


class GestureClassifier:
    """Classify a single hand's pose into one of the supported gestures."""

    SUPPORTED = (
        "thumbs_up",
        "thumbs_down",
        "open_palm",
        "fist",
        "peace",
        "ok",
        "pointing",
        "call_me",
    )

    # -- finger-state primitives -------------------------------------------
    def finger_extension(self, landmarks, finger: str) -> float:
        """Return a [0,1] "how extended" score for one non-thumb finger.

        We measure the interior angle at the PIP joint (MCP-PIP-TIP). Straight
        finger -> ~180 deg -> score ~1; curled finger -> small angle -> score 0.
        """
        mcp, pip, _dip, tip = FINGER_JOINTS[finger]
        a = angle(to_xy(landmarks[mcp]), to_xy(landmarks[pip]), to_xy(landmarks[tip]))
        return _ramp(a, CURLED_ANGLE, EXTENDED_ANGLE)

    def thumb_extension(self, landmarks) -> float:
        """[0,1] score for the thumb being straight (angle at the IP joint)."""
        a = angle(
            to_xy(landmarks[THUMB_MCP]),
            to_xy(landmarks[THUMB_IP]),
            to_xy(landmarks[THUMB_TIP]),
        )
        return _ramp(a, CURLED_ANGLE, THUMB_EXT_ANGLE)

    def count_fingers(self, landmarks) -> int:
        """Count extended fingers (0-5) using a 0.5 decision point on scores."""
        count = sum(
            1 for f in FINGER_JOINTS if self.finger_extension(landmarks, f) >= 0.5
        )
        if self.thumb_extension(landmarks) >= 0.5:
            count += 1
        return count

    # -- helpers ------------------------------------------------------------
    def _finger_scores(self, landmarks) -> Dict[str, float]:
        return {f: self.finger_extension(landmarks, f) for f in FINGER_JOINTS}

    def _thumb_points_up(self, landmarks) -> float:
        """1.0 when the thumb tip is well *above* the wrist (screen y grows down).

        Uses vertical separation normalized by palm length; ~0 when level,
        ~1 when the thumb is a full palm-length above the wrist.
        """
        scale = hand_scale(landmarks)
        dy = (to_xy(landmarks[WRIST])[1] - to_xy(landmarks[THUMB_TIP])[1]) / scale
        return _ramp(dy, 0.2, 0.9)

    def _thumb_points_down(self, landmarks) -> float:
        scale = hand_scale(landmarks)
        dy = (to_xy(landmarks[THUMB_TIP])[1] - to_xy(landmarks[WRIST])[1]) / scale
        return _ramp(dy, 0.2, 0.9)

    # -- one method per gesture --------------------------------------------
    # Each returns a confidence in [0, 1]; combining sub-scores with min()
    # enforces "all conditions must hold", while the soft ramps keep it graded.

    def score_thumbs_up(self, landmarks) -> float:
        """Thumb extended & pointing up, all four fingers curled."""
        fingers = self._finger_scores(landmarks)
        curled = min(1.0 - s for s in fingers.values())
        return min(self.thumb_extension(landmarks),
                   self._thumb_points_up(landmarks),
                   curled)

    def score_thumbs_down(self, landmarks) -> float:
        """Thumb extended & pointing down, all four fingers curled."""
        fingers = self._finger_scores(landmarks)
        curled = min(1.0 - s for s in fingers.values())
        return min(self.thumb_extension(landmarks),
                   self._thumb_points_down(landmarks),
                   curled)

    def score_open_palm(self, landmarks) -> float:
        """All five digits extended."""
        fingers = self._finger_scores(landmarks)
        return min(min(fingers.values()), self.thumb_extension(landmarks))

    def score_fist(self, landmarks) -> float:
        """All four fingers curled and thumb not sticking up/down/out."""
        fingers = self._finger_scores(landmarks)
        fingers_curled = min(1.0 - s for s in fingers.values())
        thumb_curled = 1.0 - self.thumb_extension(landmarks)
        return min(fingers_curled, thumb_curled)

    def score_peace(self, landmarks) -> float:
        """Index + middle extended and spread; ring + pinky curled.

        The spread check (angle between the two extended fingertips at the
        wrist) separates a genuine "V" from two fingers held together.
        """
        f = self._finger_scores(landmarks)
        up = min(f["index"], f["middle"])
        down = min(1.0 - f["ring"], 1.0 - f["pinky"])
        spread_angle = angle(
            to_xy(landmarks[INDEX_TIP]),
            to_xy(landmarks[WRIST]),
            to_xy(landmarks[MIDDLE_TIP]),
        )
        spread = _ramp(spread_angle, 5.0, 20.0)
        return min(up, down, spread)

    def score_ok(self, landmarks) -> float:
        """Thumb tip touches index tip (a ring); other three fingers extended."""
        f = self._finger_scores(landmarks)
        scale = hand_scale(landmarks)
        tip_gap = distance(to_xy(landmarks[THUMB_TIP]),
                           to_xy(landmarks[INDEX_TIP])) / scale
        # Smaller gap -> higher score (descending ramp).
        touching = _ramp(tip_gap, OK_TOUCH_DIST, 0.05)
        others_up = min(f["middle"], f["ring"], f["pinky"])
        return min(touching, others_up)

    def score_pointing(self, landmarks) -> float:
        """Index finger extended alone; middle/ring/pinky curled.

        Thumb position is deliberately excluded from the score (it's free to
        rest anywhere) so a relaxed "gun"-style point and a tucked-thumb point
        both register the same.
        """
        f = self._finger_scores(landmarks)
        curled_others = min(1.0 - f["middle"], 1.0 - f["ring"], 1.0 - f["pinky"])
        return min(f["index"], curled_others)

    def score_call_me(self, landmarks) -> float:
        """"Call me" / shaka: thumb + pinky extended, index/middle/ring curled."""
        f = self._finger_scores(landmarks)
        curled_middle = min(1.0 - f["index"], 1.0 - f["middle"], 1.0 - f["ring"])
        return min(self.thumb_extension(landmarks), f["pinky"], curled_middle)

    # -- dispatch -----------------------------------------------------------
    # Note: "rock / paper / scissors" aren't separate methods — they're the
    # same hand shapes as fist / open_palm / peace respectively, so those
    # gestures already cover that set under their more general names.
    def scores(self, landmarks) -> Dict[str, float]:
        """Return every gesture's confidence for this hand."""
        return {
            "thumbs_up": self.score_thumbs_up(landmarks),
            "thumbs_down": self.score_thumbs_down(landmarks),
            "open_palm": self.score_open_palm(landmarks),
            "fist": self.score_fist(landmarks),
            "peace": self.score_peace(landmarks),
            "ok": self.score_ok(landmarks),
            "pointing": self.score_pointing(landmarks),
            "call_me": self.score_call_me(landmarks),
        }

    def classify(self, landmarks, min_confidence: float = 0.6) -> Gesture:
        """Return the best-matching gesture, or ('none', best_score) if weak."""
        all_scores = self.scores(landmarks)
        name, score = max(all_scores.items(), key=lambda kv: kv[1])
        if score < min_confidence:
            return Gesture("none", score)
        return Gesture(name, score)
