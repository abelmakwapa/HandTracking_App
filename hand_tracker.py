"""
hand_tracker.py
================

Thin, opinionated wrapper around MediaPipe's **Tasks API** HandLandmarker.

Note on API version: MediaPipe removed the old ``mp.solutions.hands`` /
``mp.solutions.drawing_utils`` API from its PyPI wheels (as of the 0.10.30+
releases available at the time of writing, that legacy module is gone
entirely — only the new Tasks API ships). This wrapper targets that Tasks API
and requires a downloaded model file (see ``hand_landmarker.task`` /
README "Install" section for the fetch command).

Responsibilities:
  * own the ``HandLandmarker`` task and its lifecycle,
  * convert a BGR OpenCV frame into a clean list of per-hand results,
  * expose handedness ("Left"/"Right") + its confidence,
  * draw landmarks (drawn manually with OpenCV, since ``drawing_utils`` no
    longer exists — we source the connection topology from
    ``vision.HandLandmarksConnections``).

It deliberately knows *nothing* about gestures or interactions — those live in
their own modules so this class stays a reusable MediaPipe adapter.
"""

import time
from dataclasses import dataclass
from typing import List

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from geometry import bounding_box, palm_center

# Hand-skeleton connections, sourced from the Tasks API itself (index pairs
# into the 21-landmark list) rather than hardcoded, so it stays in sync with
# whatever topology MediaPipe ships.
_HAND_CONNECTIONS = [
    (c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS
]


@dataclass
class _Landmark:
    """Minimal (x, y, z) stand-in so downstream code (geometry.py, gesture
    classifier) can keep using attribute access (`lm.x`, `lm.y`) exactly as
    it did with the old solutions API's NormalizedLandmark objects.
    """

    x: float
    y: float
    z: float = 0.0


@dataclass
class HandResult:
    """Everything downstream code needs about a single detected hand.

    Attributes:
        label:      "Left" or "Right" (as reported by MediaPipe, from the
                    camera's point of view — see the note in the README about
                    mirrored webcam feeds).
        score:      handedness classification confidence in [0, 1].
        landmarks:  list of 21 _Landmark objects (index-matches MediaPipe's
                    canonical hand landmark ordering).
    """

    label: str
    score: float
    landmarks: list  # list of _Landmark (indexable 0..20)

    @property
    def palm_center(self):
        return palm_center(self.landmarks)

    @property
    def bbox(self):
        return bounding_box(self.landmarks)


class HandTracker:
    """MediaPipe Tasks HandLandmarker adapter.

    Use as a context manager so the underlying task graph is always closed::

        with HandTracker(model_path="hand_landmarker.task") as tracker:
            hands = tracker.process(frame)
    """

    def __init__(
        self,
        model_path: str = "hand_landmarker.task",
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 0,  # kept for CLI-compat; Tasks API has no equivalent knob
    ):
        try:
            base_options = BaseOptions(model_asset_path=model_path)
        except Exception as exc:
            raise SystemExit(
                f"[error] Could not load hand-landmark model at {model_path!r}: {exc}\n"
                f"        Download it with:\n"
                f"        curl -sL -o {model_path} "
                f"https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                f"hand_landmarker/float16/1/hand_landmarker.task"
            )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.perf_counter()

    # -- context manager plumbing -------------------------------------------
    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self._landmarker.close()

    # -- core API -----------------------------------------------------------
    def process(self, frame_bgr) -> List[HandResult]:
        """Run detection on one BGR frame and return a list of HandResults.

        The VIDEO running mode requires a monotonically increasing timestamp
        (ms) per frame, which we derive from a wall clock started at task
        creation.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.perf_counter() - self._start_time) * 1000)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hands: List[HandResult] = []
        if not result.hand_landmarks:
            return hands

        for idx, lm_list in enumerate(result.hand_landmarks):
            label, score = "Unknown", 0.0
            if idx < len(result.handedness):
                cat = result.handedness[idx][0]
                label, score = cat.category_name, cat.score
            landmarks = [_Landmark(p.x, p.y, p.z) for p in lm_list]
            hands.append(HandResult(label=label, score=score, landmarks=landmarks))
        return hands

    def draw(self, frame_bgr, hand: HandResult) -> None:
        """Overlay the landmark skeleton for one hand onto the frame.

        Drawn manually with OpenCV primitives since the Tasks API does not
        ship a `drawing_utils` equivalent.
        """
        h, w = frame_bgr.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmarks]

        for start, end in _HAND_CONNECTIONS:
            cv2.line(frame_bgr, pts[start], pts[end], (255, 255, 255), 2)
        for x, y in pts:
            cv2.circle(frame_bgr, (x, y), 4, (0, 200, 0), -1)
