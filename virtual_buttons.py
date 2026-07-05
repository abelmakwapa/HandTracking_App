"""
virtual_buttons.py
==================

On-screen virtual buttons operated by fingertip hover + pinch "click".

A button activates when, in the same frame, a hand's INDEX fingertip is
inside the button's rectangle *and* that hand performs a pinch (thumb tip
close to index tip). The pinch is edge-triggered per hand — the fingers must
separate before the same hand can press again — and each button has a short
cooldown so a lingering pinch can't machine-gun presses.

Buttons are defined in the gesture config file (see gesture_config.json):

    "buttons": [
        {"name": "hello", "label": "Say Hi", "rect": [0.05, 0.35, 0.28, 0.50]}
    ]

``rect`` is [x0, y0, x1, y1] in normalized frame coordinates. Presses are
published on the event bus as ``button.press`` {name, t}; wire an action to
one in the same config with a binding on ``"button:<name>"``.
"""

from dataclasses import dataclass
from typing import Dict, List

import cv2

from geometry import INDEX_TIP, THUMB_TIP, distance, hand_scale, to_xy

PINCH_DISTANCE = 0.4      # thumb-index gap (palm-lengths) that counts as a pinch
PRESS_COOLDOWN = 0.5      # seconds before the same button can fire again
FLASH_SECONDS = 0.25      # visual feedback duration after a press

_IDLE = (180, 120, 40)
_HOVER = (0, 200, 255)
_FLASH = (0, 255, 0)


@dataclass
class _Button:
    name: str
    label: str
    rect: tuple  # (x0, y0, x1, y1) normalized
    last_press: float = -1e9
    hovered: bool = False


class VirtualButtonPanel:
    """Owns the configured buttons; call update() per frame, then draw()."""

    def __init__(self, button_configs: List[dict], bus):
        self._bus = bus
        self._buttons: List[_Button] = []
        for cfg in button_configs:
            rect = tuple(cfg["rect"])
            if len(rect) != 4:
                raise SystemExit(
                    f"[error] Button {cfg.get('name')!r}: rect must be [x0,y0,x1,y1]."
                )
            self._buttons.append(
                _Button(name=cfg["name"], label=cfg.get("label", cfg["name"]), rect=rect)
            )
        self._was_pinched: Dict[str, bool] = {}  # per hand label

    @staticmethod
    def _inside(pt, rect) -> bool:
        x, y = pt
        x0, y0, x1, y1 = rect
        return x0 <= x <= x1 and y0 <= y <= y1

    def update(self, hands, t: float) -> None:
        """Evaluate hover + pinch for every hand against every button."""
        for b in self._buttons:
            b.hovered = False

        for hand in hands:
            tip = to_xy(hand.landmarks[INDEX_TIP])
            gap = distance(
                to_xy(hand.landmarks[THUMB_TIP]), tip
            ) / hand_scale(hand.landmarks)
            pinched = gap < PINCH_DISTANCE
            rising_edge = pinched and not self._was_pinched.get(hand.label, False)
            self._was_pinched[hand.label] = pinched

            for b in self._buttons:
                if not self._inside(tip, b.rect):
                    continue
                b.hovered = True
                if rising_edge and (t - b.last_press) > PRESS_COOLDOWN:
                    b.last_press = t
                    self._bus.publish("button.press", {"name": b.name, "t": round(t, 4)})

    def draw(self, frame, t: float) -> None:
        h, w = frame.shape[:2]
        for b in self._buttons:
            x0, y0, x1, y1 = (
                int(b.rect[0] * w), int(b.rect[1] * h),
                int(b.rect[2] * w), int(b.rect[3] * h),
            )
            if (t - b.last_press) < FLASH_SECONDS:
                color, thick = _FLASH, -1
            elif b.hovered:
                color, thick = _HOVER, 3
            else:
                color, thick = _IDLE, 2
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, thick)
            cv2.putText(
                frame, b.label, (x0 + 8, (y0 + y1) // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
