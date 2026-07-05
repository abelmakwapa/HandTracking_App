"""
cursor_controller.py
=====================

Optional OS-level mouse control driven by an index fingertip position.

This is the "a gesture becomes an actual interface action" layer: instead of
just labeling gestures on screen, it maps the index fingertip's normalized
(x, y) position to a screen coordinate and moves the real OS cursor there,
smoothed with an exponential moving average so landmark jitter doesn't make
the cursor twitch. A pinch (thumb tip near index tip) triggers a click.

This is **opt-in** (``--cursor-control`` in main.py) because it takes over the
physical mouse — a webcam glitch or a misread gesture shouldn't be able to
click things on your desktop while the app is just running in the
background.

Safety note: pyautogui's built-in fail-safe stays enabled — slamming the
physical mouse into a screen corner raises ``pyautogui.FailSafeException``.
We treat that as a deliberate "give me my mouse back" signal from the user
and pause cursor control until they re-enable it (see ``toggle()``), rather
than letting the exception crash the whole app.
"""

import time

try:
    import pyautogui

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0  # we drive this from our own frame loop timing
    _PYAUTOGUI_AVAILABLE = True
except Exception:
    _PYAUTOGUI_AVAILABLE = False

from geometry import INDEX_TIP, THUMB_TIP, distance, hand_scale, to_xy

# --- Tunable thresholds (see README "Tuning" section) -----------------------
CLICK_DISTANCE = 0.4     # thumb-index tip gap (palm-lengths) to count as a pinch/click
CLICK_COOLDOWN = 0.6     # seconds between accepted clicks, so a held pinch doesn't spam
EDGE_MARGIN_PX = 4       # keep mapped coordinates this far from screen corners, so our
                         # own moves never coincide with pyautogui's fail-safe corners


class CursorController:
    """Maps one hand's index fingertip to the OS cursor, with smoothing + click.

    Args:
        smoothing: exponential-moving-average weight given to the *new* raw
            position each frame, in (0, 1]. Lower = smoother but laggier;
            higher = snappier but jitterier. ~0.3-0.4 is a good start.
        active_region: normalized (x0, y0, x1, y1) sub-rectangle of the
            camera frame that maps to the *full* screen. Shrinking this from
            (0,0,1,1) means a comfortable hand-motion range in the middle of
            the frame can reach every screen edge, without needing to stretch
            your arm to the camera's field-of-view boundary.

    Raises:
        RuntimeError: if pyautogui isn't installed. Caught by main.py and
            turned into a SystemExit with an install hint, consistent with
            this app's other startup error handling.
    """

    def __init__(
        self,
        smoothing: float = 0.35,
        active_region=(0.15, 0.15, 0.85, 0.85),
    ):
        if not _PYAUTOGUI_AVAILABLE:
            raise RuntimeError(
                "pyautogui is not installed. Install it with `pip install pyautogui` "
                "to use --cursor-control."
            )
        self.smoothing = smoothing
        self.active_region = active_region
        self.screen_w, self.screen_h = pyautogui.size()
        self.enabled = True
        self._smoothed = None  # (x, y) in screen pixels
        self._last_click = 0.0

    def _map_to_screen(self, x_norm: float, y_norm: float):
        """Rescale a point from ``active_region`` to full screen pixels, clamped."""
        x0, y0, x1, y1 = self.active_region
        u = (x_norm - x0) / (x1 - x0)
        v = (y_norm - y0) / (y1 - y0)
        u = min(1.0, max(0.0, u))
        v = min(1.0, max(0.0, v))
        px = EDGE_MARGIN_PX + u * (self.screen_w - 2 * EDGE_MARGIN_PX)
        py = EDGE_MARGIN_PX + v * (self.screen_h - 2 * EDGE_MARGIN_PX)
        return px, py

    def update(self, landmarks) -> bool:
        """Move (and possibly click) the OS cursor from one hand's landmarks.

        Returns True if a click fired this frame (so the caller can show
        transient on-screen feedback); no-ops and returns False if control is
        currently paused (see ``enabled``/``toggle``).
        """
        if not self.enabled:
            return False

        target = self._map_to_screen(*to_xy(landmarks[INDEX_TIP]))
        if self._smoothed is None:
            self._smoothed = target
        else:
            sx, sy = self._smoothed
            tx, ty = target
            self._smoothed = (
                sx + (tx - sx) * self.smoothing,
                sy + (ty - sy) * self.smoothing,
            )

        try:
            pyautogui.moveTo(*self._smoothed)
        except pyautogui.FailSafeException:
            self._pause_from_failsafe()
            return False

        return self._maybe_click(landmarks)

    def _maybe_click(self, landmarks) -> bool:
        """Pinch (thumb tip near index tip) triggers a click, rate-limited."""
        scale = hand_scale(landmarks)
        gap = distance(to_xy(landmarks[THUMB_TIP]), to_xy(landmarks[INDEX_TIP])) / scale
        now = time.perf_counter()
        if gap >= CLICK_DISTANCE or (now - self._last_click) <= CLICK_COOLDOWN:
            return False
        try:
            pyautogui.click()
        except pyautogui.FailSafeException:
            self._pause_from_failsafe()
            return False
        self._last_click = now
        return True

    def _pause_from_failsafe(self) -> None:
        print(
            "[info] Cursor control paused: mouse hit a screen corner "
            "(fail-safe). Press 'c' to re-enable."
        )
        self.enabled = False

    def toggle(self) -> bool:
        """Flip enabled/disabled. Resets smoothing so re-enabling doesn't jump."""
        self.enabled = not self.enabled
        self._smoothed = None
        return self.enabled
