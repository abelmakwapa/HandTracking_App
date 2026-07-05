# Real-Time Hand Tracking App

Modular hand tracking with **finger counting**, **static gesture recognition**,
**dynamic (motion) gesture recognition**, **two-hand interaction detection**,
optional **OS cursor control**, an **interaction/UX layer** (virtual buttons +
configurable gesture→action bindings), **session recording/playback** for
regression testing, and **OSC/WebSocket event broadcasting** — built on
OpenCV + MediaPipe. Designed as a foundation for gesture-controlled
interfaces usable in real demos.

## File layout

| File | Responsibility |
|------|----------------|
| `geometry.py` | Pure vector/landmark math (distances, joint angles, palm center). No CV deps. |
| `fps_counter.py` | Rolling-average FPS meter. |
| `hand_tracker.py` | `HandTracker` — MediaPipe Tasks API wrapper; returns clean `HandResult`s with handedness. |
| `gesture_classifier.py` | `GestureClassifier` — one scored method per **static** (single-frame) gesture. |
| `dynamic_gestures.py` | `DynamicGestureRecognizer` — one scored method per **motion** gesture (swipe, wave, circle), driven by per-hand trajectory history. |
| `interaction_analyzer.py` | `InteractionAnalyzer` — pluggable two-hand rules (touch, pinch/zoom, active hand). |
| `cursor_controller.py` | `CursorController` — optional, maps an index fingertip to the real OS mouse (smoothed) with pinch-to-click. |
| `event_bus.py` | `EventBus` — tiny pub/sub seam decoupling the pipeline from all optional layers. |
| `smoothing.py` | `LandmarkSmoother` — configurable EMA over landmarks (anti-jitter). |
| `gesture_state.py` | `GestureStabilizer` — hold-time + hysteresis state machine; emits debounced `gesture.start`/`gesture.end` edge events. |
| `gesture_actions.py` | `GestureActionMapper` — user-editable JSON/YAML bindings from events to keystrokes/shell commands. |
| `virtual_buttons.py` | `VirtualButtonPanel` — on-screen buttons pressed by fingertip hover + pinch. |
| `recorder.py` | `SessionRecorder`/`SessionPlayer` — record raw landmark streams; replay them deterministically without a webcam. |
| `broadcaster.py` | `Broadcaster` — non-blocking OSC (UDP) / WebSocket event output for external apps. |
| `gesture_history.py` | `GestureHistory` — on-screen rolling log of recent events. |
| `gesture_config.json` | Sample config: action bindings, virtual buttons, stabilizer thresholds. |
| `gesture_config_presentation.json` | Recipe config: peace/fist/thumbs-up drive slide navigation. |
| `gesture_config_media.json` | Recipe config: OK sign play/pause, swipes skip tracks. |
| `gesture_config_accessibility.json` | Recipe config: hands-free arrow/Enter/Escape navigation + YES/NO buttons. |
| `gesture_config_game.json` | Recipe config: swipes/gestures as one-shot game inputs. |
| `whiteboard.py` | Standalone drawing app: fingertip brush, color/size buttons, pinch-erase, fist-clear. |
| `browser_client.html` | Zero-dependency web page consuming the WebSocket event/landmark stream. |
| `main.py` | CLI + `Pipeline` orchestration + overlays. |

## Architecture & data flow

```
 [webcam / video file]          [recording .jsonl]
         |                             |
     HandTracker                  SessionPlayer        <- exactly one source
    (MediaPipe)                  (no MediaPipe)
         |                             |
         +--------- raw hands ---------+
         |                                  (--record writes RAW hands here)
         v
   LandmarkSmoother (EMA, --smooth)
         v
 +--------------------- Pipeline.process (per frame) ---------------------+
 |  GestureClassifier.scores -> GestureStabilizer (hold + hysteresis)     |
 |  DynamicGestureRecognizer   (swipe/wave/circle, uses pipeline time)    |
 |  InteractionAnalyzer        (two-hand touch / pinch-zoom / active)     |
 |  VirtualButtonPanel         (hover + pinch press)                      |
 |  CursorController           (optional, live only)                      |
 +-------------------------------|----------------------------------------+
                                 |  publishes edge events
                                 v
                             EventBus
         +------------+----------+------------+-------------+
         v            v          v            v             v
  GestureHistory  ActionMapper  Broadcaster  stdout logger  (your consumer)
  (overlay log)   (keys/shell)  (OSC / WS)   (--log-events)
```

Every optional layer hangs off the `EventBus`: the pipeline publishes facts
(`gesture.start`, `dynamic`, `interaction`, `button.press`, `landmarks`) and
never references its consumers, so history, actions, broadcasting, and
logging can each be enabled/disabled independently without touching the core.

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
  For **`--cursor-control`**, macOS additionally requires granting the terminal
  app **Accessibility** permission under *System Settings → Privacy & Security →
  Accessibility* — without it, `pyautogui` silently fails to move the mouse.
- **Windows**: `pip install opencv-python mediapipe` works out of the box. If the
  webcam is slow to open, it's usually the DirectShow backend warming up.
- **Linux**: you may need `sudo apt install libgl1` for OpenCV's GUI and ensure your
  user is in the `video` group for `/dev/video0` access. For `--cursor-control`,
  `pyautogui` needs `python3-tk` and `python3-dev` (`sudo apt install python3-tk
  python3-dev`) and an X11 session (Wayland support is limited).

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

**Cursor control (opt-in — takes over your real OS mouse):**
```bash
python main.py --cursor-control
python main.py --cursor-control --cursor-hand Left --cursor-smoothing 0.25
```
Move your index fingertip to move the cursor; pinch thumb+index to click. Press
**`c`** at runtime to pause/resume without quitting. See the macOS/Linux
permission notes above — without them the cursor silently won't move.

**Gesture actions + virtual buttons (config-driven):**
```bash
python main.py --config gesture_config.json
```
Edit `gesture_config.json` to change bindings — no Python required. Buttons
appear as rectangles; hover your index fingertip inside one and pinch to press.

**Record a session, then replay it (no webcam needed):**
```bash
python main.py --record session.jsonl               # record while running live
python main.py --playback session.jsonl             # replay at real speed
python main.py --playback session.jsonl --playback-speed 0 --headless --log-events > events.log
```
The last form is the regression workflow: after a code change, re-run it and
`diff` the new `events.log` against the old one (see "Recording format &
determinism" below).

**Broadcast events to external apps:**
```bash
python main.py --osc 127.0.0.1:9000                 # OSC over UDP
python main.py --ws-port 8765                       # WebSocket server
python main.py --osc 127.0.0.1:9000 --send-landmarks  # + full landmarks per frame
```

**Stabilizer (debounce) tuning from the CLI:**
```bash
python main.py --hold-frames 6 --enter-conf 0.8 --exit-conf 0.6
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

**Cursor-control thresholds** — in `cursor_controller.py`:

| Constant / arg | Meaning |
|-----------------|---------|
| `--cursor-smoothing` (0.35) | EMA weight for the new raw position each frame; lower = smoother but laggier |
| `active_region` (0.15,0.15,0.85,0.85) | sub-rectangle of the camera frame mapped to the full screen — shrink it further if you don't want to reach the edges of the webcam's field of view to hit screen corners |
| `CLICK_DISTANCE` (0.4 palm-lengths) | thumb-index gap that counts as a pinch/click |
| `CLICK_COOLDOWN` (0.6s) | minimum time between accepted clicks, so holding a pinch doesn't spam-click |

**How to tune:** run with `--debug`, watch the live `angle`/`distance`/`score`
numbers while you perform a gesture, and nudge the constant that's on the wrong side
of your reading. E.g. if "open palm" won't register with fully-spread fingers, your
PIP angles are reading ~150° → lower `EXTENDED_ANGLE` to 150.

**Robustness thresholds:**

| Setting | Meaning |
|---------|---------|
| `--smooth` (0.5) | EMA weight of the new landmark sample; 1.0 disables, ~0.3 is very smooth but laggy. Applied *before* classification, so it also steadies gesture scores. |
| `--hold-frames` (4) | consecutive frames a gesture must persist before `gesture.start` fires |
| `--enter-conf` / `--exit-conf` (0.75 / 0.55) | hysteresis thresholds; per-gesture overrides live in the config file's `stabilizer.per_gesture` |

Note: with cursor control on, fingertip motion is smoothed twice (landmark EMA,
then the cursor's own EMA). If the cursor feels laggy, raise `--smooth` toward
1.0 or `--cursor-smoothing` toward 0.6 rather than disabling either outright.

---

## The hold-time + hysteresis state machine

Raw per-frame classification is too twitchy to drive actions — one misread
frame would fire a keystroke. `GestureStabilizer` (gesture_state.py) sits
between the classifier and the action layer, per hand:

```
IDLE ──(best gesture's score ≥ its ENTER thr)──> PENDING(count=1)

PENDING(name): each frame, look at name's own score s:
    s ≥ enter ................. count += 1
    exit ≤ s < enter .......... count += 1   (grace zone)
    s < exit .................. abort → IDLE (nothing fires)
    another gesture ≥ its enter (while s < enter)
                                → switch candidate, count restarts at 1
    count reaches hold_frames . → ACTIVE, "gesture.start" fires

ACTIVE(name): each frame:
    s ≥ exit .................. stay ACTIVE (other gestures cannot preempt)
    s < exit .................. "gesture.end" fires → IDLE (a new candidate
                                may begin PENDING in this same frame)
```

**What happens if confidence crosses the exit threshold mid-countdown?**
Hysteresis is in force *during* the hold, not just after it. The gesture
entered above `enter`, so it is only abandoned if it drops below `exit` — the
same rule an ACTIVE gesture follows. Sagging into the grace zone
(`exit ≤ s < enter`) does **not** pause or reset the countdown; it keeps
counting. Crossing below `exit` aborts the countdown entirely, the counter is
discarded, and nothing fires — no start, and no end either, because the
gesture never activated. Concretely, with enter=0.8, exit=0.6, hold=4:

```
frame:  1     2     3     4
score:  0.85  0.72  0.65  0.81   → fires on frame 4 (frames 2-3 were grace zone)
score:  0.85  0.55  0.90  0.85   → does NOT fire until frame 6 (0.55 < exit
                                    aborted; frames 3-6 must re-count from 1)
```

Two consequences worth knowing during demos: an active gesture is *sticky*
(a stronger competing gesture can't steal the slot until the active one drops
below its exit), and `gesture.end` events always carry the held duration.

---

## Recording format & determinism

**Format** — JSON Lines (`.jsonl`), UTF-8, one object per line:

```
line 1:   {"format": "handtrack-recording", "version": 1,
           "created": "<ISO-8601>", "landmarks": "raw"}
line 2+:  {"t": <seconds since session start>,
           "hands": [{"label": "Right", "score": 0.97,
                      "lm": [[x, y, z], ... 21 entries ...]}]}
```

Coordinates are **raw** (pre-smoothing) MediaPipe outputs, rounded to 5
decimals. Recording raw values means playback re-runs the *entire* downstream
pipeline — smoothing, classification, stabilizer, dynamic gestures,
interactions, buttons — exactly as live.

**Determinism guarantee** — replaying a recording produces identical event
streams (`--log-events` output) run-to-run and matching the original live
session, *given the same settings* (`--smooth`, stabilizer thresholds,
config file), because:

1. MediaPipe is bypassed on playback (landmarks come from the file) — that
   removes the only non-deterministic stage;
2. every time-based component (dynamic gestures, button cooldowns, action
   cooldowns, stabilizer, history timestamps) consumes the recorded `t`, never
   the wall clock;
3. everything else is pure float math on the same inputs.

Verified in this repo by replaying a session twice and diffing the
`--log-events` output — byte-identical. Outside the guarantee, deliberately:
the FPS overlay (display-only) and the *execution* of key/shell actions (the
decision to fire is deterministic and logged; the OS side effect obviously
isn't replayed). Cursor control is disabled during playback.

---

## Performance notes (target: 15+ FPS on a laptop)

| Feature | Cost | Mitigation already built in |
|---------|------|------------------------------|
| OSC / WebSocket sends | network I/O | all sends happen on a dedicated worker thread behind a bounded queue; overflow drops *events*, never frames |
| `--send-landmarks` | ~2-4 KB per frame | off by default; enable only when a consumer needs the full skeleton |
| EMA smoothing | 63 multiply-adds/hand | negligible |
| Stabilizer / buttons / history | dict lookups + a few putText calls | negligible; history size is bounded |
| Shell actions | process spawn | fired via `Popen` (non-blocking); a slow command can't stall the loop |
| Recording | JSON encode + buffered write | ~64 KB buffer; ~1 ms/frame. If profiling ever shows it mattering, move writes to a thread |

The heavy stage remains MediaPipe inference itself; nothing in the new layers
touches the frame pixels except the overlay drawing.

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
- [ ] Index finger extended alone (others curled) → `Pointing`
- [ ] Thumb + pinky extended, other three curled ("shaka"/phone gesture) → `Call Me`

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

**Cursor control** (`python main.py --cursor-control`)
- [ ] Top-right overlay shows `Cursor: ON (Right)`.
- [ ] Moving your right index fingertip moves the real OS mouse cursor smoothly (not jumpy).
- [ ] Pinch thumb+index together → a click registers on whatever's under the cursor; overlay flashes `Click!`.
- [ ] Holding the pinch does **not** repeat-click faster than roughly once per 0.6s (cooldown).
- [ ] Press `c` → overlay switches to `Cursor: OFF (paused)` and moving your hand no longer moves the mouse; press `c` again to resume.
- [ ] Slam your *physical* mouse into a screen corner while cursor control is on → control pauses itself with a console message (fail-safe).
- [ ] `--cursor-control` without `pyautogui` installed → clear "pyautogui is not installed" error, no crash/traceback.

**Stabilized gestures / actions** (`--config gesture_config.json`)
- [ ] Hold a thumbs-up: the `LOCKED:` line appears only after a beat (~4 frames), not instantly.
- [ ] Let the gesture decay slowly — the lock should release cleanly, not flicker on/off.
- [ ] With the sample config: thumbs-up presses Space (put a cursor in a text field first).
- [ ] Flash a gesture for a single frame (quick flick) → no action fires.

**Virtual buttons**
- [ ] Buttons render as rectangles; hovering an index fingertip highlights one.
- [ ] Pinch while hovering → button flashes green, `BTN:` line appears in the history panel.
- [ ] Holding the pinch does not repeat-fire; release and pinch again to press again.

**Recording / playback**
- [ ] `--record s.jsonl`, wave at the camera, quit → "Recording saved (N frames)".
- [ ] `--playback s.jsonl` → your session replays as a skeleton on a black canvas, same gestures detected.
- [ ] Replay twice with `--headless --log-events --playback-speed 0` and `diff` the outputs → identical.

**Broadcast**
- [ ] `--osc 127.0.0.1:9000` + any OSC monitor → events arrive at `/handtrack/...` addresses.
- [ ] `--ws-port 8765` + a browser `new WebSocket('ws://localhost:8765/')` → JSON events arrive.
- [ ] Kill the consumer mid-run → app keeps running at full FPS (sends are off-thread).

**History panel**
- [ ] Each gesture/interaction/button event appends a timestamped line at the top-right; old lines scroll off.

**Error handling**
- [ ] `--source does_not_exist.mp4` → clear "Video file not found" message, no crash.
- [ ] `--source 9` (no such camera) → clear "Could not open webcam" message with fixes.
- [ ] Unplug/cover the webcam mid-run → tolerates drops, exits cleanly after a burst.
- [ ] `--playback not_a_recording.txt` → clear "not a hand-tracking recording" error.
- [ ] `--config` with a typo'd binding → clear config error naming the bad field, no traceback.

### Gesture pairs most likely to be confused — watch these

| Pair | Why they collide | What to watch / how to disambiguate |
|------|------------------|-------------------------------------|
| **OK sign vs. pinch-start** | Both are thumb+index tips together. Static OK requires the *other three fingers extended*; a pinch usually curls them. If your middle/ring/pinky drift down, OK score drops. | Keep the last three fingers clearly up for OK. |
| **Peace vs. open palm** | If ring/pinky don't curl enough, peace bleeds into a partial open palm. | Curl ring+pinky firmly; the `spread` term needs a real V-gap between index/middle. |
| **Fist vs. thumbs up/down** | All three share four curled fingers; only the thumb differs. A thumb that's tucked but slightly angled can read as up/down. | For a clean fist, tuck the thumb across the fingers; for thumbs up/down, extend the thumb fully and clear of the palm. |
| **Thumbs up vs. thumbs down** | Distinguished purely by thumb-tip vertical position vs. wrist. Near-horizontal thumbs are ambiguous. | Point the thumb clearly up or down, not sideways. |
| **Fist vs. OK (edge case)** | A loose fist where thumb rests near the index tip can start scoring OK. | Watch the `ok` score in `--debug`; raise `OK_TOUCH_DIST` down if false OKs appear. |
| **Pointing vs. finger-count "1"** | Both involve just the index finger extended — `Pointing` is a gesture *label*, while `count_fingers` reports it as `1` regardless of gesture name. Not a real conflict, but easy to expect them to disagree. | These are two separate features reading the same pose; seeing `Fingers: 1` and `Pointing` together is correct, not redundant. |
| **Call Me vs. thumbs up/down** | Both extend the thumb while curling other fingers; they differ only in whether the pinky is also extended. If the pinky doesn't curl/extend cleanly, the two can flicker between each other. | Curl the pinky firmly down for thumbs up/down; extend it clearly out to the side for Call Me. |
| **Pinch-to-click vs. OK sign vs. two-hand pinch/zoom** | All three key off thumb-index tip proximity, just on different hands/scopes: `cursor_controller`'s click uses *one* hand's thumb-index gap; `GestureClassifier.score_ok` uses the same gap *plus* requires the other three fingers up; `InteractionAnalyzer`'s pinch/zoom uses the distance *between two separate hands*, not within one hand. | If cursor control is on and you also want to show an OK sign, expect an accidental click each time your thumb and index meet — that's the same signal doing two jobs on purpose. Turn off `--cursor-control` if you need OK-sign-heavy interaction without stray clicks. |
| **Swipe vs. wave** | A fast single back-and-forth can look like the start of a swipe in one direction, then reverse. Since a swipe fires (and clears history) on the first qualifying displacement, a wave that starts fast may fire as a swipe instead. | Start a wave with smaller, more controlled oscillations; reserve full-arm motion for swipes. |
| **Circle vs. wave** | A wide, curved wave can accumulate enough swept angle to look "circle-ish" before completing a straight reversal. | Keep circles round and at a consistent radius from a fixed center; keep waves flat (mostly x-motion, `WAVE_MAX_Y_DRIFT` guards this). |
| **Active-hand vs. swipe on the *other* hand** | If one hand swipes, `rule_active_hand` (in `interaction_analyzer.py`) will also likely name it "active" that frame — both are correct simultaneously, not a bug, but can look redundant on screen. | Expected overlap; the two systems answer different questions (which hand moved vs. what shape did the motion trace). |

---

## Recipes: turning this into an actual app

Each recipe below uses only what's already built — either a ready-made config
file or a documented flag combination. Sample configs (`gesture_config_*.json`)
are starting points; copy and edit them rather than editing `gesture_config.json`
in place.

**1. Gesture-controlled presentations** — peace = next slide, fist = previous,
thumbs up = start slideshow:
```bash
python main.py --config gesture_config_presentation.json
```
Click into your slideshow app first so the keystrokes land there (F5 starts
most slideshow apps in presenter mode; `b` blanks the screen in many).

**2. Music/media player control** — swipe to skip, OK sign to play/pause:
```bash
python main.py --config gesture_config_media.json
```
Media-key bindings vary by app/OS; edit the `"keys"` arrays in the config to
match your player's actual shortcuts (e.g. swap `space` for `k` on YouTube).

**3. Drawing/painting app** — index fingertip as brush, on-screen swatches for
color/size, pinch to erase, hold a fist to clear:
```bash
python whiteboard.py
python whiteboard.py --source 1 --smooth 0.3   # smoother strokes, different camera
```
This is a separate entry point ([whiteboard.py](whiteboard.py)), not a
`main.py --config` mode — it needs a canvas that persists across frames, which
`main.py`'s per-frame overlay isn't built for. It reuses `HandTracker`,
`VirtualButtonPanel`, `GestureStabilizer`, and `LandmarkSmoother` directly.
Press `s` to save a PNG snapshot.

**4. Game control** — swipes for movement, gestures for actions:
```bash
python main.py --config gesture_config_game.json
```
**Limitation to know going in:** swipes and the stabilizer fire *one*
keypress per motion/hold, not a held key — good for turn-based games or
platformer "step" input, not for a character that should keep walking while
your hand stays up. For continuous movement, don't route through key events
at all: write a small loop that polls `pipeline.stabilizer.active_gesture("Right")`
directly (see `gesture_state.py`) and hold your own key/controller state for
as long as that returns non-`None`.

**5. Virtual whiteboard** — same as #3; "drawing app" and "whiteboard" are the
same feature here. Use `whiteboard.py`.

**6. Accessibility tool** — hands-free navigation for limited keyboard access:
```bash
python main.py --config gesture_config_accessibility.json
```
Thumbs up/down = up/down arrow, open palm = Enter, fist = Escape, swipes =
Tab/Shift+Tab, plus two large on-screen YES/NO buttons. The longer
`hold_frames` (6) and lower thresholds in this config trade a bit of latency
for fewer accidental activations — tune `--hold-frames`/`--enter-conf` further
based on the user's actual hand control.

**7. Live demo tool** — show detection happening in real time:
```bash
python main.py --config gesture_config.json --debug
```
The `LOCKED:` label only appears once the hold-time countdown completes (so
the audience sees the debounce working, not raw flicker), the top-right
history panel timestamps every event, and `--debug` exposes the raw per-frame
scores so you can explain *why* a gesture did or didn't fire.

**8. AR/VR app controller** — broadcast to Processing/TouchDesigner/Unity:
```bash
python main.py --osc 127.0.0.1:9000 --send-landmarks
```
Point your OSC listener at `/handtrack/gesture/start`, `/handtrack/dynamic`,
`/handtrack/interaction`, `/handtrack/landmarks`, etc. — see the OSC wire
format documented in [broadcaster.py](broadcaster.py)'s module docstring.

**9. Streaming overlay / tutorial recordings** — capture a session once,
replay it identically for editing or teaching:
```bash
python main.py --record demo_take1.jsonl --config gesture_config.json
python main.py --playback demo_take1.jsonl                # re-watch it
python main.py --playback demo_take1.jsonl --osc 127.0.0.1:9000  # re-broadcast it to OBS/TouchDesigner
```
Because playback is deterministic (see above), a recorded tutorial always
reproduces the same detected gestures — useful for a fixed voiceover.

**10. Ambient UI** — on-screen elements that react to hand presence:
```bash
python main.py --config gesture_config.json
```
The sample config's two buttons (`gesture_config.json`) already demonstrate
this — they highlight on hover before any click. For "reacts to presence
alone" (no click needed), read `hand.bbox`/`hand.palm_center` overlap with a
region yourself in a small script, the same way `VirtualButtonPanel._inside`
does, but skip its pinch-edge check.

**11. Browser-based gesture input** — a web page reacting to your hands:
```bash
python main.py --ws-port 8765 --send-landmarks
```
Then open [browser_client.html](browser_client.html) in any browser (no
server needed — plain `file://` works, or serve it with `python -m
http.server`). It renders live landmark dots on a canvas and logs every
gesture/interaction/button event; use it as the starting point for a browser
game or web AR overlay.

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
- **New cursor action:** `CursorController.update()` currently only moves + clicks;
  a right-click or scroll could be added the same way `_maybe_click` works — read a
  geometric signal (e.g. a fist while moving = drag) and call the matching `pyautogui`
  function, gated by its own cooldown.
