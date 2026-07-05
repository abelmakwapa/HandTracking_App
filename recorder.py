"""
recorder.py
===========

Record and replay raw landmark sessions — webcam-free regression testing.

File format (documented guarantee)
----------------------------------
JSON Lines (one JSON object per line, UTF-8):

    Line 1 (header):
        {"format": "handtrack-recording", "version": 1,
         "created": "<ISO-8601>", "landmarks": "raw"}

    Every following line (one per frame):
        {"t": <float seconds since session start>,
         "hands": [{"label": "Left"|"Right", "score": <float>,
                    "lm": [[x, y, z], ...21 entries...]}, ...]}

Coordinates are the *raw* MediaPipe outputs (pre-smoothing), rounded to 5
decimals, in normalized [0,1] units. Recording raw (not smoothed) landmarks
means playback re-runs the ENTIRE downstream pipeline — smoothing,
classification, stabilization, dynamic gestures, interactions — exactly as a
live run would.

Determinism guarantee
---------------------
Replaying a recording produces identical gesture/action event streams to
each other and to the original live session, **given the same settings**
(--smooth alpha, stabilizer thresholds/hold frames, config file), because:

  * MediaPipe is bypassed on playback (landmarks come from the file), which
    removes the only non-deterministic stage;
  * every time-dependent component (dynamic gestures, button cooldowns,
    stabilizer, history) consumes the pipeline timestamp ``t`` from the
    file, never the wall clock;
  * everything else is pure float math on the landmark values.

So the regression workflow is: record once; after a code change, replay with
``--playback session.jsonl --headless --log-events > new.log`` and diff
against the previous log. Exceptions that are *deliberately* outside the
guarantee: the FPS overlay (display-only) and OS side effects of actions
(the *decision* to fire is deterministic; pyautogui/shell execution is not
replayed into the log). Cursor control is disabled during playback.
"""

import json
from datetime import datetime, timezone
from typing import Iterator, List, Tuple

from hand_tracker import HandResult, _Landmark

FORMAT_NAME = "handtrack-recording"
FORMAT_VERSION = 1


class SessionRecorder:
    """Writes one JSONL line per frame. Buffered; call close() (or use main's
    normal shutdown path) to flush."""

    def __init__(self, path: str):
        self._fh = open(path, "w", encoding="utf-8", buffering=1 << 16)
        header = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "landmarks": "raw",
        }
        self._fh.write(json.dumps(header) + "\n")
        self.frames = 0

    def write_frame(self, t: float, hands: List[HandResult]) -> None:
        rec = {
            "t": round(t, 4),
            "hands": [
                {
                    "label": h.label,
                    "score": round(h.score, 4),
                    "lm": [[round(p.x, 5), round(p.y, 5), round(p.z, 5)] for p in h.landmarks],
                }
                for h in hands
            ],
        }
        self._fh.write(json.dumps(rec) + "\n")
        self.frames += 1

    def close(self) -> None:
        self._fh.close()
        print(f"[info] Recording saved ({self.frames} frames).")


class SessionPlayer:
    """Iterates (t, hands) frames from a recording, validating the header."""

    def __init__(self, path: str):
        try:
            self._fh = open(path, "r", encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"[error] Cannot open recording {path!r}: {exc}")
        try:
            header = json.loads(self._fh.readline())
        except json.JSONDecodeError:
            raise SystemExit(f"[error] {path!r} is not a valid recording (bad header).")
        if header.get("format") != FORMAT_NAME:
            raise SystemExit(
                f"[error] {path!r} is not a hand-tracking recording "
                f"(format={header.get('format')!r})."
            )
        if header.get("version", 0) > FORMAT_VERSION:
            print(
                f"[warn] Recording version {header['version']} is newer than this "
                f"app supports ({FORMAT_VERSION}); attempting playback anyway."
            )

    def __iter__(self) -> Iterator[Tuple[float, List[HandResult]]]:
        for line_no, line in enumerate(self._fh, start=2):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] Skipping corrupt frame at line {line_no}.")
                continue
            hands = [
                HandResult(
                    label=h["label"],
                    score=h["score"],
                    landmarks=[_Landmark(x, y, z) for x, y, z in h["lm"]],
                )
                for h in rec.get("hands", [])
            ]
            yield rec["t"], hands
        self._fh.close()
