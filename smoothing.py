"""
smoothing.py
============

Exponential-moving-average (EMA) smoothing of hand landmarks.

MediaPipe landmarks jitter by a pixel or two even for a perfectly still hand.
That jitter propagates into joint angles and tip distances, which makes
gesture confidences flicker and the cursor twitch. An EMA is the cheapest
effective fix:

    smoothed = previous * (1 - alpha) + raw * alpha

``alpha`` is the weight of the *new* sample: 1.0 disables smoothing entirely,
lower values are smoother but laggier. 0.4-0.6 is a good range at ~30 FPS;
below ~0.3 the hand visibly trails its skeleton.

The smoother is keyed by hand label and resets when a hand leaves the frame
so a re-entering hand snaps to its new position instead of gliding across
the screen from where it disappeared.

Determinism note: given the same input landmark sequence and the same alpha,
the output sequence is identical — this is what allows recorded sessions
(recorder.py) to replay bit-identically through the pipeline.
"""

from typing import Dict, List

from hand_tracker import _Landmark


class LandmarkSmoother:
    """Per-hand EMA over all 21 landmarks (x, y, z)."""

    def __init__(self, alpha: float = 0.5):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("smoothing alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: Dict[str, List[_Landmark]] = {}

    def prune_absent(self, present_labels) -> None:
        """Forget hands that left the frame (prevents re-entry gliding)."""
        for label in list(self._state):
            if label not in present_labels:
                del self._state[label]

    def smooth(self, label: str, landmarks: list) -> list:
        """Return a smoothed copy of one hand's landmark list."""
        prev = self._state.get(label)
        if prev is None or len(prev) != len(landmarks):
            smoothed = [_Landmark(lm.x, lm.y, lm.z) for lm in landmarks]
        else:
            a = self.alpha
            smoothed = [
                _Landmark(
                    p.x + (lm.x - p.x) * a,
                    p.y + (lm.y - p.y) * a,
                    p.z + (lm.z - p.z) * a,
                )
                for p, lm in zip(prev, landmarks)
            ]
        self._state[label] = smoothed
        return smoothed
