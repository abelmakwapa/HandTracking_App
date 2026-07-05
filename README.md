# Real-Time Hand Tracking App

Modular hand tracking with **finger counting**, **static gesture recognition**,
**dynamic (motion) gesture recognition**, and **two-hand interaction detection**,
built on OpenCV + MediaPipe. Designed as a foundation for gesture-controlled
interfaces, not a one-off demo.

## File layout

| File | Responsibility |
|------|----------------|
| `geometry.py` | Pure vector/landmark math (distances, joint angles, palm center). No CV deps. |
| `fps_counter.py` | Rolling-average FPS meter. |
| `hand_tracker.py` | `HandTracker` — MediaPipe Tasks API wrapper; returns clean `HandResult`s with handedness. |
| `gesture_classifier.py` | `GestureClassifier` — one scored method per **static** (single-frame) gesture. |
| `dynamic_gestures.py` | `DynamicGestureRecognizer` — one scored method per **motion** gesture (swipe, wave, circle), driven by per-hand trajectory history. |
| `interaction_analyzer.py` | `InteractionAnalyzer` — pluggable two-hand rules (touch, pinch/zoom, active hand). |
| `main.py` | CLI + orchestration + on-screen overlay. |

---

## (a) Install

MediaPipe currently supports **Python 3.9–3.11** (avoid 3.12+ for now).

```bash
cd HandTracking_App
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Download the hand-landmark model** (required — current MediaPipe releases no
longer bundle the legacy `mp.solutions` API or its model internally; the app uses
the newer Tasks API, which loads the model from disk):

```bash
curl -sL -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

`main.py` looks for `hand_landmarker.task` in the working directory by default;
override with `--model-path` if you put it elsewhere. If it's missing, the app
exits with this same download command rather than crashing.

**OS-specific notes**

- **macOS**: the first webcam run triggers a camera-permission prompt. Grant the
  *terminal app* camera access under *System Settings → Privacy & Security →
  Camera*, then **restart the terminal** (not just the script) for it to take
  effect. Apple-silicon Macs work on native `python.org`/Homebrew Python 3.11.
- **Windows**: `pip install opencv-python mediapipe` works out of the box. If the
  webcam is slow to open, it's usually the DirectShow backend warming up.
- **Linux**: you may need `sudo apt install libgl1` for OpenCV's GUI and ensure your
  user is in the `video` group for `/dev/video0` access.

---

## (b) Run

**Webcam (default, index 0):**
```bash
python main.py
```

**Different webcam / window size:**
```bash
python main.py --source 1 --width 1280 --height 720
```

**Video file:**
```bash
python main.py --source /path/to/video.mov
```

**Debug overlay (raw coords + all gesture scores + motion/interaction deltas):**
```bash
python main.py --debug
```

**Tune detection / performance:**
```bash
python main.py --det-conf 0.7 --track-conf 0.5 --gesture-conf 0.65 --complexity 0
```

Press **`q`** in the window to quit. Full flag list: `python main.py --help`.

> **Performance:** `--complexity 0` (the default, "lite" model) is what gets you
> 15+ FPS on a laptop CPU. Use `--complexity 1` only if you have GPU/CPU headroom
> and want slightly steadier landmarks.

> **Handedness note:** the webcam feed is mirrored (`cv2.flip`) so it feels natural.
> MediaPipe labels handedness from the *image's* perspective, which after mirroring
> matches your real left/right hand. On a raw (unmirrored) video file the labels are
> flipped — that's expected.

---

## (c) Thresholds & tuning

All gesture geometry is **scale-normalized** by palm length (`hand_scale`), so
thresholds hold whether your hand is near or far from the camera. Scores are built
from *soft ramps* (`_ramp`) rather than hard cutoffs, so a gesture that's *almost*
right scores ~0.9 instead of flipping on/off.

**Gesture thresholds** — in `gesture_classifier.py`:

| Constant | Meaning | Raise it to… | Lower it to… |
|----------|---------|--------------|--------------|
| `EXTENDED_ANGLE` (160°) | PIP joint angle above which a finger is "straight" | demand straighter fingers | count slightly-bent fingers as extended |
| `CURLED_ANGLE` (100°) | angle below which a finger is "curled" | — | — |
| `THUMB_EXT_ANGLE` (150°) | IP angle for the thumb being straight | require a straighter thumb | — |
| `OK_TOUCH_DIST` (0.35 palm-lengths) | max thumb–index tip gap to register an OK "ring" | require tips closer together | tolerate a looser ring |

Finger counting uses a **0.5 decision point** on the per-finger extension score
(`count_fingers`). Each gesture combines sub-scores with `min(...)` = "all
conditions must hold"; the weakest condition sets the confidence.

**Dynamic (motion) gesture thresholds** — in `dynamic_gestures.py`:

| Constant | Meaning |
|----------|---------|
| `HISTORY_SECONDS` (1.5s) | how much trajectory each hand retains; raises the ceiling for how slow a wave/circle can be and still register |
| `SWIPE_WINDOW` (0.5s) | a swipe's start-to-end displacement must happen within this time |
| `SWIPE_MIN_DISTANCE` (0.25 frame-widths) | net displacement needed to call it a swipe (not just drift) |
| `SWIPE_AXIS_RATIO` (1.8x) | how much the dominant axis must beat the other axis, to reject diagonal motion |
| `WAVE_MIN_REVERSALS` (3) | x-direction changes required within the window to call it a wave |
| `WAVE_MIN_AMPLITUDE` (0.04) | minimum x movement per half-cycle before it counts as a reversal (filters hand jitter) |
| `WAVE_MAX_Y_DRIFT` (0.15) | if the hand drifts vertically more than this during the window, it's not a wave |
| `CIRCLE_MIN_ANGLE` (300°) | cumulative swept angle required to call it a full circle |
| `CIRCLE_MIN_RADIUS` (0.05 frame-widths) | minimum average radius from the trajectory's centroid, to reject in-place jitter |
| `GESTURE_COOLDOWN` (0.8s) | suppresses re-firing the same motion gesture on the same hand immediately after it fires |

Each motion gesture returns a confidence too (e.g. swipe confidence scales with how
far past `SWIPE_MIN_DISTANCE` the displacement got), visible in `--debug` next to the
gesture name.

**Interaction thresholds** — in `interaction_analyzer.py`:

| Constant | Meaning |
|----------|---------|
| `TOUCH_DISTANCE` (0.12) | normalized-frame distance between palm centers to call it a touch/clap |
| `PINCH_DELTA` (0.015) | per-window change in inter-hand distance to fire zoom-in/out |
| `InteractionAnalyzer(history=8)` | frames of history; larger = smoother zoom, more lag |

**How to tune:** run with `--debug`, watch the live `angle`/`distance`/`score`
numbers while you perform a gesture, and nudge the constant that's on the wrong side
of your reading. E.g. if "open palm" won't register with fully-spread fingers, your
PIP angles are reading ~150° → lower `EXTENDED_ANGLE` to 150.

---

## (d) Manual validation checklist

Since webcam input can't be unit-tested easily, verify each feature by hand. Run
`python main.py --debug` and check the overlay.

**Core tracking**
- [ ] Landmarks (skeleton) track your hand smoothly as you move it.
- [ ] Top-left **FPS** reads ≥15 and is stable (not wildly jumping per frame).
- [ ] **Hands: N** matches how many hands are visible.

**Handedness**
- [ ] Right hand shows `Right (0.9x)`, left shows `Left (0.9x)`, with a plausible score.

**Finger counting** (hold each pose, read `Fingers: N`)
- [ ] Fist → **0**
- [ ] Index only → **1**
- [ ] Peace (index+middle) → **2**
- [ ] Three fingers → **3**
- [ ] Open palm → **5**

**Gestures** (read the label + score, expect ≥0.6)
- [ ] Thumbs up → `Thumbs Up`
- [ ] Thumbs down → `Thumbs Down`
- [ ] Open palm, fingers spread → `Open Palm`
- [ ] Closed fist → `Fist`
- [ ] V / peace sign, fingers apart → `Peace`
- [ ] Thumb+index circle, other fingers up → `OK Sign`

**Dynamic (motion) gestures** (single hand; watch the orange line above the interaction row)
- [ ] Swipe your hand quickly left-to-right across the frame → `[swipe] right` (or `Left: Swipe Right` in the overlay).
- [ ] Swipe right-to-left → `Swipe Left`; swipe down→up → `Swipe Up`; up→down → `Swipe Down`.
- [ ] Wave your hand side-to-side 3+ times (like waving hello) → `Wave`.
- [ ] Draw a full circle in the air with your hand → `Circle (CW)` or `Circle (CCW)` depending on direction.
- [ ] After any motion gesture fires, it shouldn't immediately re-fire on the next frame (0.8s cooldown).

**Two-hand interactions** (need both hands in frame)
- [ ] Bring palms together / clap → `[touch]` appears at the bottom.
- [ ] Hold both hands up, move them apart → `[pinch_zoom] zoom_in`; together → `zoom_out`.
- [ ] Wiggle one hand while holding the other still → `[active_hand]` names the moving one.

**Error handling**
- [ ] `--source does_not_exist.mp4` → clear "Video file not found" message, no crash.
- [ ] `--source 9` (no such camera) → clear "Could not open webcam" message with fixes.
- [ ] Unplug/cover the webcam mid-run → tolerates drops, exits cleanly after a burst.

### Gesture pairs most likely to be confused — watch these

| Pair | Why they collide | What to watch / how to disambiguate |
|------|------------------|-------------------------------------|
| **OK sign vs. pinch-start** | Both are thumb+index tips together. Static OK requires the *other three fingers extended*; a pinch usually curls them. If your middle/ring/pinky drift down, OK score drops. | Keep the last three fingers clearly up for OK. |
| **Peace vs. open palm** | If ring/pinky don't curl enough, peace bleeds into a partial open palm. | Curl ring+pinky firmly; the `spread` term needs a real V-gap between index/middle. |
| **Fist vs. thumbs up/down** | All three share four curled fingers; only the thumb differs. A thumb that's tucked but slightly angled can read as up/down. | For a clean fist, tuck the thumb across the fingers; for thumbs up/down, extend the thumb fully and clear of the palm. |
| **Thumbs up vs. thumbs down** | Distinguished purely by thumb-tip vertical position vs. wrist. Near-horizontal thumbs are ambiguous. | Point the thumb clearly up or down, not sideways. |
| **Fist vs. OK (edge case)** | A loose fist where thumb rests near the index tip can start scoring OK. | Watch the `ok` score in `--debug`; raise `OK_TOUCH_DIST` down if false OKs appear. |
| **Swipe vs. wave** | A fast single back-and-forth can look like the start of a swipe in one direction, then reverse. Since a swipe fires (and clears history) on the first qualifying displacement, a wave that starts fast may fire as a swipe instead. | Start a wave with smaller, more controlled oscillations; reserve full-arm motion for swipes. |
| **Circle vs. wave** | A wide, curved wave can accumulate enough swept angle to look "circle-ish" before completing a straight reversal. | Keep circles round and at a consistent radius from a fixed center; keep waves flat (mostly x-motion, `WAVE_MAX_Y_DRIFT` guards this). |
| **Active-hand vs. swipe on the *other* hand** | If one hand swipes, `rule_active_hand` (in `interaction_analyzer.py`) will also likely name it "active" that frame — both are correct simultaneously, not a bug, but can look redundant on screen. | Expected overlap; the two systems answer different questions (which hand moved vs. what shape did the motion trace). |

---

## Extending it

- **New static gesture:** add a `score_<name>()` method to `GestureClassifier`, register it
  in `scores()`. Return a `min(...)` of soft-ramped sub-conditions.
- **New dynamic (motion) gesture:** add a `_check_<name>(hist, now, label)` method to
  `DynamicGestureRecognizer` following the pattern in `_check_swipe`/`_check_wave`/
  `_check_circle` (read `hist.positions`, return a `DynamicGestureEvent` or `None`, call
  `self._fire(hist, now, "<name>")` on match), then add it to the check tuple in `update()`.
- **New two-hand combo:** write a `rule_<name>(ctx)` function in
  `interaction_analyzer.py` returning an `InteractionEvent` or `None`, then append it
  to the `RULES` list. No loop changes needed.
