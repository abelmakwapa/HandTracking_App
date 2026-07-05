"""
main.py
=======

Orchestrates the hand-tracking pipeline.

Architecture / data flow (text diagram):

    [webcam / video file]        [recording .jsonl]
            |                           |
        HandTracker                SessionPlayer          <- exactly one source
       (MediaPipe)                (no MediaPipe)
            |                           |
            +--------- raw hands -------+
            |                                     --record writes RAW hands here
            v
      LandmarkSmoother (EMA, --smooth)            <- robustness layer
            v
    +---------------------- Pipeline.process (per frame) -----------------+
    |  GestureClassifier.scores  ->  GestureStabilizer (hold + hysteresis)|
    |  DynamicGestureRecognizer  (swipe/wave/circle, uses pipeline time)  |
    |  InteractionAnalyzer       (two-hand touch / pinch-zoom / active)   |
    |  VirtualButtonPanel        (hover + pinch press)                    |
    |  CursorController          (optional, live only)                    |
    +------------------------------|---------------------------------
                                   | publishes edge events
                                   v
                               EventBus
            +------------+---------+-----------+------------+
            v            v         v           v            v
     GestureHistory  ActionMapper  Broadcaster  stdout      (yours here)
     (overlay log)   (keys/shell)  (OSC / WS)   (--log-events)

Every consumer hangs off the EventBus, so each layer can be enabled/disabled
independently without touching the pipeline. Run ``python main.py --help``
for all options.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from cursor_controller import CursorController
from dynamic_gestures import DynamicGestureRecognizer
from event_bus import EventBus
from fps_counter import FPSCounter
from gesture_actions import GestureActionMapper, load_config
from gesture_classifier import GestureClassifier
from gesture_history import GestureHistory
from gesture_state import GestureStabilizer
from geometry import THUMB_TIP, to_xy
from hand_tracker import HandTracker
from interaction_analyzer import InteractionAnalyzer
from recorder import SessionPlayer, SessionRecorder
from smoothing import LandmarkSmoother
from virtual_buttons import VirtualButtonPanel

# Pretty labels for on-screen display.
GESTURE_LABELS = {
    "thumbs_up": "Thumbs Up",
    "thumbs_down": "Thumbs Down",
    "open_palm": "Open Palm",
    "fist": "Fist",
    "peace": "Peace",
    "ok": "OK Sign",
    "pointing": "Pointing",
    "call_me": "Call Me",
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

PLAYBACK_CANVAS = (480, 640)  # (h, w) synthetic frame for --playback display


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
        "--det-conf", type=float, default=0.6,
        help="MediaPipe min detection confidence [0-1].",
    )
    p.add_argument(
        "--track-conf", type=float, default=0.5,
        help="MediaPipe min tracking confidence [0-1].",
    )
    p.add_argument(
        "--gesture-conf", type=float, default=0.6,
        help="Min confidence to display a raw gesture label [0-1].",
    )
    p.add_argument(
        "--complexity", type=int, choices=(0, 1), default=0,
        help="Unused by the Tasks-API landmarker; kept for CLI compatibility.",
    )
    p.add_argument(
        "--model-path", default="hand_landmarker.task",
        help="Path to the MediaPipe HandLandmarker .task model file.",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Overlay raw landmark coords and gesture metrics (angles/distances).",
    )

    ux = p.add_argument_group("interaction / UX layer")
    ux.add_argument(
        "--config", default=None,
        help="Gesture config file (.json or .yaml): action bindings, virtual "
             "buttons, stabilizer threshold overrides. See gesture_config.json.",
    )
    ux.add_argument(
        "--hold-frames", type=int, default=4,
        help="Consecutive frames a gesture must persist before its start event "
             "fires (config file 'stabilizer' section overrides this).",
    )
    ux.add_argument(
        "--enter-conf", type=float, default=0.75,
        help="Stabilizer enter threshold (config file overrides).",
    )
    ux.add_argument(
        "--exit-conf", type=float, default=0.55,
        help="Stabilizer exit threshold; must be < enter (config file overrides).",
    )

    rb = p.add_argument_group("robustness")
    rb.add_argument(
        "--smooth", type=float, default=0.5,
        help="EMA landmark smoothing: weight of the new sample in (0,1]. "
             "1.0 disables smoothing; lower = smoother but laggier.",
    )
    rb.add_argument(
        "--record", default=None, metavar="PATH",
        help="Record the raw landmark stream to a .jsonl file for later playback.",
    )
    rb.add_argument(
        "--playback", default=None, metavar="PATH",
        help="Replay a recorded .jsonl session instead of opening a camera. "
             "Deterministic given the same settings (see README).",
    )
    rb.add_argument(
        "--playback-speed", type=float, default=1.0,
        help="Playback pacing multiplier; 0 = no pacing (as fast as possible, "
             "for regression runs).",
    )
    rb.add_argument(
        "--headless", action="store_true",
        help="No display window (use with --playback for automated regression runs).",
    )

    out = p.add_argument_group("output / integration")
    out.add_argument(
        "--osc", default=None, metavar="HOST:PORT",
        help="Broadcast events as OSC-over-UDP to this target (e.g. 127.0.0.1:9000).",
    )
    out.add_argument(
        "--ws-port", type=int, default=None,
        help="Serve events to WebSocket clients on this port (needs 'websockets').",
    )
    out.add_argument(
        "--send-landmarks", action="store_true",
        help="Also broadcast the full landmark set every frame (higher bandwidth).",
    )
    out.add_argument(
        "--log-events", action="store_true",
        help="Print every pipeline event as a JSON line on stdout (regression diffing).",
    )
    out.add_argument(
        "--no-history", action="store_true",
        help="Disable the on-screen gesture history panel.",
    )

    cur = p.add_argument_group("cursor control")
    cur.add_argument(
        "--cursor-control", action="store_true",
        help="Drive the real OS cursor from an index fingertip (pinch to click). "
             "Off by default since it takes over your mouse. Toggle with 'c'.",
    )
    cur.add_argument(
        "--cursor-hand", choices=("Left", "Right"), default="Right",
        help="Which hand's index fingertip drives the cursor.",
    )
    cur.add_argument(
        "--cursor-smoothing", type=float, default=0.35,
        help="EMA smoothing weight for cursor movement (lower = smoother, laggier).",
    )
    return p.parse_args(argv)


def open_capture(source: str) -> cv2.VideoCapture:
    """Open a webcam index or a video file, with actionable error messages."""
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


def draw_hand_overlay(frame, hand, best_name, best_score, stable_name,
                      finger_count, scores, debug):
    """Per-hand labels: handedness, fingers, raw best gesture, locked gesture."""
    h, w = frame.shape[:2]
    x0, y0, _x1, _y1 = hand.bbox
    px, py = int(x0 * w), int(y0 * h)
    py = max(py, 80)  # keep text on-screen

    color = GREEN if hand.label == "Right" else YELLOW
    _text(frame, f"{hand.label} ({hand.score:.2f})", (px, py - 60), color)
    _text(frame, f"Fingers: {finger_count}", (px, py - 40), color)
    _text(frame, f"{GESTURE_LABELS.get(best_name, best_name)} {best_score:.2f}",
          (px, py - 20), color)
    if stable_name:
        # The debounced/hysteresis-stabilized gesture — what actions fire on.
        _text(frame, f"LOCKED: {GESTURE_LABELS.get(stable_name, stable_name)}",
              (px, py), WHITE)

    if debug:
        tx, ty = to_xy(hand.landmarks[THUMB_TIP])
        _text(frame, f"thumb_tip=({tx:.2f},{ty:.2f})", (px, py + 20), WHITE, 0.45, 1)
        for i, (name, sc) in enumerate(sorted(scores.items())):
            _text(frame, f"{name}:{sc:.2f}", (px, py + 40 + i * 16), WHITE, 0.45, 1)


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
    """Bottom-anchored, upward-growing stack of (text, color) lines."""
    h, _w = frame.shape[:2]
    y = h - 20 - (len(items) - 1) * 24 if items else h - 20
    for text, color in items:
        _text(frame, text, (10, y), color, 0.6, 2)
        y += 24


# ---------------------------------------------------------------------------
# Pipeline: the shared per-frame path for live AND playback modes
# ---------------------------------------------------------------------------
class Pipeline:
    """Everything between "here are this frame's hands" and "here's the frame
    to display". Both the live loop and the playback loop call process(),
    which is what makes replays representative of live behavior.
    """

    def __init__(self, args, bus: EventBus, stabilizer: GestureStabilizer,
                 smoother, buttons, history, cursor):
        self.args = args
        self.bus = bus
        self.classifier = GestureClassifier()
        self.analyzer = InteractionAnalyzer()
        self.dynamic = DynamicGestureRecognizer()
        self.stabilizer = stabilizer
        self.smoother = smoother          # LandmarkSmoother or None
        self.buttons = buttons            # VirtualButtonPanel or None
        self.history = history            # GestureHistory or None
        self.cursor = cursor              # CursorController or None
        self.fps = FPSCounter(window=30)
        self._prev_interactions = set()   # for edge-deduping interaction events
        self._clicked = False

    def process(self, frame, hands, t: float, draw_fn) -> None:
        """Run one frame through every enabled stage and draw overlays.

        Args:
            frame: BGR image to draw on (camera frame or blank playback canvas).
            hands: HandResults for this frame (from MediaPipe or a recording).
            t: pipeline timestamp, seconds. Live: wall clock since start.
               Playback: the recorded timestamp — this is the determinism seam.
            draw_fn: callable(frame, hand) that renders the skeleton (the
                tracker's drawer live, or the local one in playback).
        """
        present = {h.label for h in hands}

        # -- robustness: EMA smoothing before anything consumes landmarks ----
        if self.smoother is not None:
            self.smoother.prune_absent(present)
            for hand in hands:
                hand.landmarks = self.smoother.smooth(hand.label, hand.landmarks)

        self.dynamic.prune_absent(present)
        for topic, payload in self.stabilizer.prune(present, t):
            self.bus.publish(topic, payload)

        dynamic_events = []
        for hand in hands:
            draw_fn(frame, hand)

            scores = self.classifier.scores(hand.landmarks)
            best_name, best_score = max(scores.items(), key=lambda kv: kv[1])
            if best_score < self.args.gesture_conf:
                best_name = "none"

            for topic, payload in self.stabilizer.update(hand.label, scores, t):
                self.bus.publish(topic, payload)

            fingers = self.classifier.count_fingers(hand.landmarks)
            draw_hand_overlay(
                frame, hand, best_name, best_score,
                self.stabilizer.active_gesture(hand.label),
                fingers, scores, self.args.debug,
            )

            dyn = self.dynamic.update(hand.label, hand.palm_center, now=t)
            if dyn is not None:
                dynamic_events.append(dyn)
                self.bus.publish("dynamic", {
                    "hand": dyn.hand_label, "name": dyn.name, "detail": dyn.detail,
                    "confidence": round(dyn.confidence, 4), "t": round(t, 4),
                })

        interactions = self.analyzer.analyze(hands)
        current = set()
        for ev in interactions:
            current.add((ev.name, ev.detail))
            # Interactions re-fire every frame while held; publish edges only
            # so subscribers (actions/history/broadcast) see one event per
            # occurrence, not 30/second.
            if (ev.name, ev.detail) not in self._prev_interactions:
                self.bus.publish("interaction", {
                    "name": ev.name, "detail": ev.detail,
                    "value": round(ev.value, 4), "t": round(t, 4),
                })
        self._prev_interactions = current

        if self.buttons is not None:
            self.buttons.update(hands, t)
            self.buttons.draw(frame, t)

        self._clicked = False
        if self.cursor is not None:
            target = next((h for h in hands if h.label == self.args.cursor_hand), None)
            if target is not None:
                self._clicked = self.cursor.update(target.landmarks)

        if self.args.send_landmarks:
            self.bus.publish("landmarks", {
                "t": round(t, 4),
                "hands": [
                    {"label": h.label,
                     "lm": [[round(p.x, 4), round(p.y, 4)] for p in h.landmarks]}
                    for h in hands
                ],
            })

        # -- overlays ---------------------------------------------------------
        items = [(format_dynamic_event(ev, self.args.debug), ORANGE)
                 for ev in dynamic_events]
        for ev in interactions:
            text = f"[{ev.name}] {ev.detail}"
            if self.args.debug:
                text += f" ({ev.value:+.3f})"
            items.append((text, RED))
        draw_event_stack(frame, items)

        self.fps.tick()
        _text(frame, f"FPS: {self.fps.fps:4.1f}", (10, 30), GREEN, 0.7, 2)
        _text(frame, f"Hands: {len(hands)}", (10, 55), GREEN, 0.6, 2)

        if self.cursor is not None:
            w = frame.shape[1]
            status = "ON" if self.cursor.enabled else "OFF (paused)"
            _text(frame, f"Cursor: {status} ({self.args.cursor_hand})",
                  (w - 340, 30), ORANGE, 0.6, 2)
            if self._clicked:
                _text(frame, "Click!", (w - 340, 55), ORANGE, 0.6, 2)

        if self.history is not None:
            self.history.draw(frame)

    def handle_key(self, key: int) -> bool:
        """Returns False when the app should quit."""
        if key == ord("q"):
            return False
        if key == ord("c") and self.cursor is not None:
            state = self.cursor.toggle()
            print(f"[info] Cursor control {'enabled' if state else 'disabled'}.")
        return True


# ---------------------------------------------------------------------------
# Setup + the two frame-source loops
# ---------------------------------------------------------------------------
def build_pipeline(args) -> Pipeline:
    config = load_config(args.config) if args.config else {}
    bus = EventBus()

    stab_cfg = config.get("stabilizer", {})
    stabilizer = GestureStabilizer(
        enter=stab_cfg.get("enter", args.enter_conf),
        exit=stab_cfg.get("exit", args.exit_conf),
        hold_frames=stab_cfg.get("hold_frames", args.hold_frames),
        per_gesture=stab_cfg.get("per_gesture"),
    )

    smoother = LandmarkSmoother(args.smooth) if args.smooth < 1.0 else None
    history = None if args.no_history else GestureHistory(bus)
    buttons = (
        VirtualButtonPanel(config["buttons"], bus) if config.get("buttons") else None
    )
    if config.get("bindings"):
        GestureActionMapper(config, bus)  # lives via its bus subscriptions

    if args.osc or args.ws_port:
        from broadcaster import Broadcaster
        try:
            Broadcaster(bus, osc_target=args.osc, ws_port=args.ws_port)
        except RuntimeError as exc:
            raise SystemExit(f"[error] {exc}")

    if args.log_events:
        def log(topic):
            return lambda p: print(
                json.dumps({"event": topic, **p}, sort_keys=True), flush=True
            )
        for topic in ("gesture.start", "gesture.end", "dynamic",
                      "interaction", "button.press"):
            bus.subscribe(topic, log(topic))

    cursor = None
    if args.cursor_control:
        if args.playback:
            print("[info] --cursor-control is disabled during playback.")
        else:
            try:
                cursor = CursorController(smoothing=args.cursor_smoothing)
            except RuntimeError as exc:
                raise SystemExit(f"[error] {exc}")
            print(
                f"[info] Cursor control enabled, driven by the {args.cursor_hand} "
                f"hand. Press 'c' to pause/resume; pinch thumb+index to click."
            )

    return Pipeline(args, bus, stabilizer, smoother, buttons, history, cursor)


def _show(args, pipeline, frame) -> bool:
    """Display + key handling; returns False to quit. No-op in --headless."""
    if args.headless:
        return True
    display = cv2.resize(frame, (args.width, args.height))
    cv2.imshow("Hand Tracking", display)
    return pipeline.handle_key(cv2.waitKey(1) & 0xFF)


def run_playback(args, pipeline: Pipeline) -> int:
    """Replay a recorded session through the pipeline (no camera, no MediaPipe)."""
    if args.record:
        print("[warn] --record is ignored during playback.")
    print(f"[info] Replaying {args.playback!r}"
          + (" (headless)" if args.headless else " — press 'q' to quit."))

    # Local skeleton drawer (the live one lives on HandTracker, which we
    # deliberately don't construct here).
    from hand_tracker import _HAND_CONNECTIONS

    def draw_skeleton(frame, hand):
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmarks]
        for a, b in _HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0, 200, 0), -1)

    prev_t = None
    for t, hands in SessionPlayer(args.playback):
        if args.playback_speed > 0 and prev_t is not None:
            time.sleep(max(0.0, (t - prev_t) / args.playback_speed))
        prev_t = t

        frame = np.zeros((*PLAYBACK_CANVAS, 3), dtype=np.uint8)
        pipeline.process(frame, hands, t, draw_skeleton)
        if not _show(args, pipeline, frame):
            break

    if not args.headless:
        cv2.destroyAllWindows()
    print("[info] Playback finished.")
    return 0


def run_live(args, pipeline: Pipeline) -> int:
    if not os.path.exists(args.model_path):
        raise SystemExit(
            f"[error] Hand-landmark model not found at {args.model_path!r}.\n"
            f"        Download it with:\n"
            f"        curl -sL -o {args.model_path} "
            f"https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            f"hand_landmarker/float16/1/hand_landmarker.task"
        )

    cap = open_capture(args.source)
    recorder = SessionRecorder(args.record) if args.record else None
    is_file = not args.source.isdigit()
    consecutive_drops = 0
    MAX_DROPS = 30  # tolerate transient webcam hiccups before giving up
    t0 = time.perf_counter()

    print("[info] Running. Press 'q' in the window to quit.")
    try:
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

                # Mirror the webcam so it feels natural (not for files).
                if not is_file:
                    frame = cv2.flip(frame, 1)

                t = time.perf_counter() - t0
                try:
                    hands = tracker.process(frame)
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[warn] Detection failed on a frame, skipping: {exc}")
                    continue

                # Record RAW landmarks (pre-smoothing) so playback re-runs the
                # full pipeline — see recorder.py's determinism notes.
                if recorder is not None:
                    recorder.write_frame(t, hands)

                pipeline.process(frame, hands, t, tracker.draw)
                if not _show(args, pipeline, frame):
                    break
    finally:
        if recorder is not None:
            recorder.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    pipeline = build_pipeline(args)
    try:
        if args.playback:
            return run_playback(args, pipeline)
        return run_live(args, pipeline)
    except KeyboardInterrupt:
        print("\n[info] Interrupted by user.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
