"""
main.py
=======

Orchestrates the hand-tracking pipeline:

    frame -> HandTracker -> [ GestureClassifier per hand ]     (static pose)
                          -> [ DynamicGestureRecognizer per hand ] (motion)
                          -> InteractionAnalyzer (two hands)
                          -> overlay + display

Run ``python main.py --help`` for all options.
"""

import argparse
import sys
import time

import cv2

from dynamic_gestures import DynamicGestureRecognizer
from fps_counter import FPSCounter
from gesture_classifier import GestureClassifier
from geometry import THUMB_TIP, to_xy
from hand_tracker import HandTracker
from interaction_analyzer import InteractionAnalyzer

# Pretty labels for on-screen display.
GESTURE_LABELS = {
    "thumbs_up": "Thumbs Up",
    "thumbs_down": "Thumbs Down",
    "open_palm": "Open Palm",
    "fist": "Fist",
    "peace": "Peace",
    "ok": "OK Sign",
    "none": "-",
}

# Pretty labels for dynamic (motion) gestures, keyed by (name, detail).
DYNAMIC_GESTURE_LABELS = {
    ("swipe", "left"): "Swipe Left",
    ("swipe", "right"): "Swipe Right",
    ("swipe", "up"): "Swipe Up",
    ("swipe", "down"): "Swipe Down",
    ("circle", "clockwise"): "Circle (CW)",
    ("circle", "counterclockwise"): "Circle (CCW)",
}

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
ORANGE = (0, 140, 255)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time hand tracking with gesture & interaction detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        default="0",
        help="Input source: a webcam index (e.g. 0) or a path to a video file.",
    )
    p.add_argument("--width", type=int, default=960, help="Display window width.")
    p.add_argument("--height", type=int, default=540, help="Display window height.")
    p.add_argument(
        "--max-hands", type=int, default=2, help="Maximum number of hands to track."
    )
    p.add_argument(
        "--det-conf",
        type=float,
        default=0.6,
        help="MediaPipe min detection confidence [0-1].",
    )
    p.add_argument(
        "--track-conf",
        type=float,
        default=0.5,
        help="MediaPipe min tracking confidence [0-1].",
    )
    p.add_argument(
        "--gesture-conf",
        type=float,
        default=0.6,
        help="Min confidence to accept a gesture label [0-1].",
    )
    p.add_argument(
        "--complexity",
        type=int,
        choices=(0, 1),
        default=0,
        help="Unused by the Tasks-API landmarker; kept for CLI compatibility.",
    )
    p.add_argument(
        "--model-path",
        default="hand_landmarker.task",
        help="Path to the MediaPipe HandLandmarker .task model file.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Overlay raw landmark coords and gesture metrics (angles/distances).",
    )
    return p.parse_args(argv)


def open_capture(source: str) -> cv2.VideoCapture:
    """Open a webcam index or a video file, with actionable error messages.

    Returns an opened VideoCapture, or raises SystemExit with guidance.
    """
    is_index = source.isdigit()
    if is_index:
        # CAP_AVFOUNDATION is the correct backend on macOS and avoids a noisy
        # fallback; on other platforms OpenCV picks a sensible default.
        cap = (
            cv2.VideoCapture(int(source), cv2.CAP_AVFOUNDATION)
            if sys.platform == "darwin"
            else cv2.VideoCapture(int(source))
        )
    else:
        import os

        if not os.path.exists(source):
            raise SystemExit(
                f"[error] Video file not found: {source!r}\n"
                f"        Check the path, or pass a webcam index like --source 0."
            )
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        if is_index:
            raise SystemExit(
                f"[error] Could not open webcam index {source}.\n"
                f"        - Is another app using the camera? Close it and retry.\n"
                f"        - On macOS, grant camera permission to your terminal in\n"
                f"          System Settings > Privacy & Security > Camera.\n"
                f"        - Try a different index (--source 1)."
            )
        raise SystemExit(
            f"[error] Could not open video file {source!r} (unsupported codec?).\n"
            f"        Try converting it to H.264 MP4, e.g. with ffmpeg."
        )
    return cap


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------
def _text(img, s, org, color=WHITE, scale=0.6, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_hand_overlay(frame, hand, gesture, finger_count, classifier, debug):
    """Draw per-hand label, gesture, finger count, and (in debug) raw metrics."""
    h, w = frame.shape[:2]
    x0, y0, _x1, _y1 = hand.bbox
    px, py = int(x0 * w), int(y0 * h)
    py = max(py, 60)  # keep text on-screen

    color = GREEN if hand.label == "Right" else YELLOW
    _text(frame, f"{hand.label} ({hand.score:.2f})", (px, py - 40), color)
    _text(frame, f"Fingers: {finger_count}", (px, py - 20), color)
    _text(
        frame,
        f"{GESTURE_LABELS.get(gesture.name, gesture.name)} {gesture.score:.2f}",
        (px, py),
        color,
    )

    if debug:
        # Raw landmark coords for a couple of key points + all gesture scores.
        tx, ty = to_xy(hand.landmarks[THUMB_TIP])
        _text(
            frame,
            f"thumb_tip=({tx:.2f},{ty:.2f})",
            (px, py + 20),
            WHITE,
            0.45,
            1,
        )
        scores = classifier.scores(hand.landmarks)
        for i, (name, sc) in enumerate(sorted(scores.items())):
            _text(
                frame,
                f"{name}:{sc:.2f}",
                (px, py + 40 + i * 16),
                WHITE,
                0.45,
                1,
            )


def format_dynamic_event(ev, debug):
    """Human-readable label for a fired DynamicGestureEvent."""
    pretty = DYNAMIC_GESTURE_LABELS.get((ev.name, ev.detail))
    if pretty is None:
        pretty = "Wave" if ev.name == "wave" else f"{ev.name} {ev.detail}"
    text = f"{ev.hand_label}: {pretty}"
    if debug:
        text += f" ({ev.confidence:.2f})"
    return text


def draw_event_stack(frame, items):
    """Draw a bottom-anchored, upward-growing stack of (text, color) lines.

    Used for both two-hand interaction events and per-hand dynamic gesture
    events so they share one non-overlapping layout instead of two
    independently positioned overlays.
    """
    h, _w = frame.shape[:2]
    y = h - 20 - (len(items) - 1) * 24 if items else h - 20
    for text, color in items:
        _text(frame, text, (10, y), color, 0.6, 2)
        y += 24


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(args) -> int:
    import os

    if not os.path.exists(args.model_path):
        raise SystemExit(
            f"[error] Hand-landmark model not found at {args.model_path!r}.\n"
            f"        Download it with:\n"
            f"        curl -sL -o {args.model_path} "
            f"https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            f"hand_landmarker/float16/1/hand_landmarker.task"
        )

    cap = open_capture(args.source)
    fps = FPSCounter(window=30)
    classifier = GestureClassifier()
    analyzer = InteractionAnalyzer()
    dynamic_recognizer = DynamicGestureRecognizer()

    is_file = not args.source.isdigit()
    consecutive_drops = 0
    MAX_DROPS = 30  # tolerate transient webcam hiccups before giving up

    print("[info] Running. Press 'q' in the window to quit.")

    with HandTracker(
        model_path=args.model_path,
        max_hands=args.max_hands,
        min_detection_confidence=args.det_conf,
        min_tracking_confidence=args.track_conf,
        model_complexity=args.complexity,
    ) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_file:
                    print("[info] End of video file.")
                    break
                # Webcam: a dropped frame is usually transient.
                consecutive_drops += 1
                if consecutive_drops > MAX_DROPS:
                    print(
                        "[error] Lost the webcam feed "
                        f"({MAX_DROPS} consecutive dropped frames). Exiting."
                    )
                    break
                time.sleep(0.01)
                continue
            consecutive_drops = 0

            # Mirror the webcam so it feels like a mirror (natural for the user).
            if not is_file:
                frame = cv2.flip(frame, 1)

            try:
                hands = tracker.process(frame)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[warn] Detection failed on a frame, skipping: {exc}")
                continue

            dynamic_recognizer.prune_absent({h.label for h in hands})
            dynamic_events = []
            for hand in hands:
                tracker.draw(frame, hand)
                gesture = classifier.classify(
                    hand.landmarks, min_confidence=args.gesture_conf
                )
                fingers = classifier.count_fingers(hand.landmarks)
                draw_hand_overlay(frame, hand, gesture, fingers, classifier, args.debug)

                dyn_event = dynamic_recognizer.update(hand.label, hand.palm_center)
                if dyn_event is not None:
                    dynamic_events.append(dyn_event)

            events = analyzer.analyze(hands)

            overlay_items = [(format_dynamic_event(ev, args.debug), ORANGE) for ev in dynamic_events]
            for ev in events:
                text = f"[{ev.name}] {ev.detail}"
                if args.debug:
                    text += f" ({ev.value:+.3f})"
                overlay_items.append((text, RED))
            draw_event_stack(frame, overlay_items)

            fps.tick()
            _text(frame, f"FPS: {fps.fps:4.1f}", (10, 30), GREEN, 0.7, 2)
            _text(frame, f"Hands: {len(hands)}", (10, 55), GREEN, 0.6, 2)

            display = cv2.resize(frame, (args.width, args.height))
            cv2.imshow("Hand Tracking", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
