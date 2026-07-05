"""
whiteboard.py
=============

A standalone drawing/whiteboard app built on the same components as main.py:
index fingertip = brush, on-screen buttons pick color/size, pinch = eraser,
fist = clear the canvas.

This is a separate entry point (not a main.py mode) because it needs its own
persistent drawing surface that survives across frames — main.py's frame is
redrawn from scratch every loop, which is wrong for a canvas you paint onto
over time. Everything it reuses (HandTracker, VirtualButtonPanel, EventBus,
LandmarkSmoother) is imported straight from the existing modules; no
duplicated gesture logic.

Controls:
    Index fingertip touching nothing special -> draws a line as it moves.
    Pinch (thumb+index close)                -> eraser (draws background color).
    Fist (held ~0.5s, via the stabilizer)    -> clears the whole canvas.
    On-screen swatches / size buttons        -> pinch-press to select.
    'q'                                       -> quit.  's'  -> save PNG.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from event_bus import EventBus
from gesture_classifier import GestureClassifier
from gesture_state import GestureStabilizer
from geometry import INDEX_TIP, THUMB_TIP, distance, hand_scale, to_xy
from hand_tracker import HandTracker
from smoothing import LandmarkSmoother
from virtual_buttons import VirtualButtonPanel

PINCH_DISTANCE = 0.4  # thumb-index gap (palm-lengths) that counts as "eraser active"

COLOR_BUTTONS = [
    {"name": "color_black", "label": "",  "rect": [0.90, 0.03, 0.98, 0.11], "color": (30, 30, 30)},
    {"name": "color_red",   "label": "",  "rect": [0.90, 0.13, 0.98, 0.21], "color": (0, 0, 220)},
    {"name": "color_green", "label": "",  "rect": [0.90, 0.23, 0.98, 0.31], "color": (0, 180, 0)},
    {"name": "color_blue",  "label": "",  "rect": [0.90, 0.33, 0.98, 0.41], "color": (220, 100, 0)},
]
SIZE_BUTTONS = [
    {"name": "size_small", "label": "S", "rect": [0.90, 0.46, 0.98, 0.54], "size": 4},
    {"name": "size_med",   "label": "M", "rect": [0.90, 0.56, 0.98, 0.64], "size": 9},
    {"name": "size_large", "label": "L", "rect": [0.90, 0.66, 0.98, 0.74], "size": 16},
]
BACKGROUND = (250, 250, 250)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Hand-tracking whiteboard / drawing app.")
    p.add_argument("--source", default="0", help="Webcam index or video file path.")
    p.add_argument("--model-path", default="hand_landmarker.task")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--smooth", type=float, default=0.4,
                   help="EMA landmark smoothing; lower = smoother brush strokes.")
    p.add_argument("--save-dir", default=".", help="Where 's' saves PNG snapshots.")
    return p.parse_args(argv)


def open_capture(source: str) -> cv2.VideoCapture:
    is_index = source.isdigit()
    cap = (
        cv2.VideoCapture(int(source), cv2.CAP_AVFOUNDATION)
        if is_index and sys.platform == "darwin"
        else cv2.VideoCapture(int(source) if is_index else source)
    )
    if not cap.isOpened():
        raise SystemExit(
            f"[error] Could not open {'webcam ' + source if is_index else source!r}. "
            "See main.py's --help notes on camera permissions."
        )
    return cap


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.model_path):
        raise SystemExit(
            f"[error] Model not found at {args.model_path!r}. See README 'Install'."
        )

    cap = open_capture(args.source)
    bus = EventBus()
    classifier = GestureClassifier()
    stabilizer = GestureStabilizer(enter=0.75, exit=0.55, hold_frames=8)  # fist=clear needs a deliberate hold
    smoother = LandmarkSmoother(args.smooth)
    buttons = VirtualButtonPanel(COLOR_BUTTONS + SIZE_BUTTONS, bus)

    state = {"color": (30, 30, 30), "size": 9}

    def pick_color(name, color):
        bus.subscribe("button.press", lambda p, n=name, c=color:
                       state.update(color=c) if p["name"] == n else None)

    def pick_size(name, size):
        bus.subscribe("button.press", lambda p, n=name, s=size:
                       state.update(size=s) if p["name"] == n else None)

    for b in COLOR_BUTTONS:
        pick_color(b["name"], b["color"])
    for b in SIZE_BUTTONS:
        pick_size(b["name"], b["size"])

    cleared = {"flag": False}
    bus.subscribe("gesture.start", lambda p: cleared.update(flag=True)
                  if p["name"] == "fist" else None)

    canvas = None  # allocated on first frame once we know its size
    prev_point = {}  # per-hand last brush position, for line segments

    print("[info] Whiteboard running. Draw with your index finger, pinch to erase, "
          "hold a fist to clear. 's' saves, 'q' quits.")
    t0 = time.perf_counter()
    try:
        with HandTracker(model_path=args.model_path, max_hands=2) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[info] End of input.")
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                if canvas is None:
                    canvas = np.full((h, w, 3), BACKGROUND, dtype=np.uint8)

                t = time.perf_counter() - t0
                try:
                    hands = tracker.process(frame)
                except Exception as exc:
                    print(f"[warn] Detection failed on a frame, skipping: {exc}")
                    continue

                present = {hd.label for hd in hands}
                smoother.prune_absent(present)
                for hd in hands:
                    hd.landmarks = smoother.smooth(hd.label, hd.landmarks)
                for label in list(prev_point):
                    if label not in present:
                        del prev_point[label]

                for _topic, payload in stabilizer.prune(present, t):
                    pass  # fist-clear only cares about start, handled below
                for hd in hands:
                    scores = classifier.scores(hd.landmarks)
                    for topic, payload in stabilizer.update(hd.label, scores, t):
                        bus.publish(topic, payload)

                buttons.update(hands, t)

                if cleared["flag"]:
                    canvas[:] = BACKGROUND
                    cleared["flag"] = False
                    prev_point.clear()

                for hd in hands:
                    tip = to_xy(hd.landmarks[INDEX_TIP])
                    px, py = int(tip[0] * w), int(tip[1] * h)
                    gap = distance(to_xy(hd.landmarks[THUMB_TIP]), tip) / hand_scale(hd.landmarks)
                    erasing = gap < PINCH_DISTANCE
                    color = BACKGROUND if erasing else state["color"]
                    size = state["size"] * 3 if erasing else state["size"]

                    # Skip drawing while the fingertip is over the button rail,
                    # so picking a color doesn't leave a stroke behind it.
                    over_buttons = tip[0] > 0.88
                    if not over_buttons:
                        prev = prev_point.get(hd.label)
                        if prev is not None:
                            cv2.line(canvas, prev, (px, py), color, size, cv2.LINE_AA)
                        else:
                            cv2.circle(canvas, (px, py), size // 2, color, -1)
                        prev_point[hd.label] = (px, py)
                    else:
                        prev_point.pop(hd.label, None)

                display = canvas.copy()
                buttons.draw(display, t)
                for hd in hands:
                    tip = to_xy(hd.landmarks[INDEX_TIP])
                    cv2.circle(display, (int(tip[0] * w), int(tip[1] * h)),
                               max(3, state["size"] // 2), (0, 0, 0), 1)
                cv2.putText(display, f"Color/Size selectors ->", (int(0.60 * w), 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

                out = cv2.resize(display, (args.width, args.height))
                cv2.imshow("Whiteboard", out)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    path = os.path.join(args.save_dir, f"whiteboard_{int(time.time())}.png")
                    cv2.imwrite(path, canvas)
                    print(f"[info] Saved {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
