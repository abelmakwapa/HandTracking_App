"""
geometry.py
===========

Low-level vector / landmark math shared by the rest of the app.

MediaPipe returns 21 landmarks per hand as *normalized* coordinates
(x, y in [0, 1] relative to the frame width/height, z is depth relative to
the wrist). For gesture geometry we mostly work in the (x, y) plane and,
where useful, treat the normalized coordinates as a 2D vector space.

All helpers here are pure functions with no MediaPipe / OpenCV dependency so
they are trivial to reason about and (unlike the webcam pipeline) easy to
unit-test in isolation.

Landmark index reference (MediaPipe Hands):

    0  : WRIST
    1  : THUMB_CMC     2 : THUMB_MCP     3 : THUMB_IP      4 : THUMB_TIP
    5  : INDEX_MCP     6 : INDEX_PIP     7 : INDEX_DIP     8 : INDEX_TIP
    9  : MIDDLE_MCP   10 : MIDDLE_PIP   11 : MIDDLE_DIP   12 : MIDDLE_TIP
    13 : RING_MCP     14 : RING_PIP     15 : RING_DIP     16 : RING_TIP
    17 : PINKY_MCP    18 : PINKY_PIP    19 : PINKY_DIP    20 : PINKY_TIP
"""

import math
from typing import Sequence, Tuple

# ---------------------------------------------------------------------------
# Named landmark indices (avoids magic numbers scattered through the code).
# ---------------------------------------------------------------------------
WRIST = 0

THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# (mcp, pip, dip, tip) tuples for the four non-thumb fingers.
FINGER_JOINTS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

# A convenient "palm center" is the average of the wrist and the four
# finger MCP knuckles — it is stable even when fingers move.
PALM_POINTS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

Point = Tuple[float, float]


def to_xy(landmark) -> Point:
    """Extract a plain (x, y) tuple from a MediaPipe landmark object."""
    return (landmark.x, landmark.y)


def distance(a: Point, b: Point) -> float:
    """Euclidean distance between two 2D points (in normalized units)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle(a: Point, b: Point, c: Point) -> float:
    """
    Interior angle (in degrees) at vertex ``b`` formed by the segments
    b->a and b->c.

    We build two vectors originating at the joint ``b`` and use the dot-product
    definition:

        cos(theta) = (v1 . v2) / (|v1| * |v2|)

    A straight (fully extended) finger gives an angle near 180 degrees; a
    curled finger bends toward 0-90 degrees. This angle is scale-invariant,
    so it behaves the same whether the hand is near or far from the camera.
    """
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 == 0 or mag2 == 0:
        return 180.0

    # Clamp to [-1, 1] to guard against tiny floating point overshoot.
    cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_theta))


def centroid(points: Sequence[Point]) -> Point:
    """Average position of a set of points."""
    n = len(points)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


def palm_center(landmarks) -> Point:
    """Stable palm center = mean of wrist + the four finger knuckles."""
    return centroid([to_xy(landmarks[i]) for i in PALM_POINTS])


def hand_scale(landmarks) -> float:
    """
    A reference length used to normalize distances so thresholds are
    independent of how close the hand is to the camera.

    We use the wrist -> middle-finger MCP distance (the palm length), which
    stays roughly constant regardless of finger pose.
    """
    return distance(to_xy(landmarks[WRIST]), to_xy(landmarks[MIDDLE_MCP])) or 1e-6


def bounding_box(landmarks) -> Tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) of all landmarks (normalized)."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (min(xs), min(ys), max(xs), max(ys))
