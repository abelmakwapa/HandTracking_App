"""
fps_counter.py
==============

Rolling-average FPS meter.

Instantaneous FPS (``1 / delta_between_two_frames``) is extremely noisy: a
single slow frame makes the number jump around and it is useless for judging
real performance. Instead we keep a sliding window of recent frame timestamps
and derive FPS from the *span* of that window, which smooths the reading.
"""

import time
from collections import deque


class FPSCounter:
    """Sliding-window FPS estimator.

    Args:
        window: number of recent frames to average over. Larger = smoother
            but slower to react to genuine performance changes.
    """

    def __init__(self, window: int = 30):
        self._timestamps: deque = deque(maxlen=window)

    def tick(self) -> None:
        """Record that a frame was just processed."""
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """Average FPS over the current window (0.0 until warmed up)."""
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return 0.0
        # (n - 1) intervals across `span` seconds.
        return (len(self._timestamps) - 1) / span
