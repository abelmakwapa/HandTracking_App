"""
gesture_state.py
================

Hold-time + hysteresis state machine for stabilized gesture events.

GestureClassifier gives an instantaneous best guess every frame; raw, that
signal is too twitchy to drive actions — one misdetected frame would fire a
keystroke. This module turns the per-frame score stream into clean
*edge events* ("gesture X started" / "gesture X ended") using two mechanisms:

  * **Hold time**: a gesture must stay the winning candidate for N
    consecutive frames before it activates (kills single-frame misdetections).
  * **Hysteresis**: activation requires confidence >= ``enter``; once active
    (or while counting), the gesture survives until confidence drops below a
    *lower* ``exit`` threshold. The gap between the two stops rapid on/off
    flicker when confidence hovers around any single cutoff.

State machine (per hand)
------------------------

    IDLE ──(best gesture's score >= its enter thr)──> PENDING(count=1)

    PENDING(name):  each frame, look at `name`'s own score s:
        s >= enter ............ count += 1
        exit <= s < enter ..... count += 1   (grace zone — see below)
        s < exit .............. abort -> IDLE (count discarded, nothing fires)
        another gesture's score >= its enter
          while s < enter ..... switch candidate -> PENDING(other, count=1)
        count reaches hold_frames -> ACTIVE(name), publish "gesture.start"

    ACTIVE(name):   each frame, look at `name`'s own score s:
        s >= exit ............. stay ACTIVE (other gestures CANNOT preempt)
        s < exit .............. publish "gesture.end" -> IDLE
                                (entry logic runs again the same frame, so a
                                 new gesture can begin PENDING immediately)

**The exit-threshold-during-hold question:** while PENDING, hysteresis is
already in force. If confidence sags into the grace zone (exit <= s < enter)
mid-countdown, the countdown *continues* — the gesture entered above `enter`,
so it is only abandoned if it drops below `exit`, exactly like an ACTIVE
gesture would be. Only crossing below `exit` aborts the countdown, and an
aborted countdown fires nothing (the action never triggered, so there is no
"end" event either). Worked example with enter=0.8, exit=0.6, hold=4:

    frame:  1     2     3     4     5
    score:  0.85  0.72  0.65  0.81  0.83
    state:  P(1)  P(2)  P(3)  P(4)->ACTIVE, "gesture.start" fires on frame 4
                  ^^^^^^^^^^ grace zone: counting continued

    frame:  1     2     3
    score:  0.85  0.55  0.9
    state:  P(1)  IDLE  P(1)   <- 0.55 < exit aborted the count; frame 3
                                  starts over from 1. Nothing ever fired.

Determinism: all logic is frame-count + score based (no wall clock), so a
replayed recording (recorder.py) reproduces identical start/end events.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class _HandState:
    candidate: Optional[str] = None
    count: int = 0
    active: Optional[str] = None
    active_since: float = 0.0


class GestureStabilizer:
    """Turns per-frame gesture scores into debounced start/end events.

    Args:
        enter: default confidence needed to begin/continue counting a gesture.
        exit: default confidence below which a pending/active gesture is dropped.
        hold_frames: consecutive frames a candidate must survive before firing.
        per_gesture: optional {gesture_name: {"enter": x, "exit": y}} overrides,
            typically loaded from the gesture config file.
    """

    def __init__(
        self,
        enter: float = 0.75,
        exit: float = 0.55,
        hold_frames: int = 4,
        per_gesture: Optional[dict] = None,
    ):
        if exit >= enter:
            raise ValueError("hysteresis requires exit threshold < enter threshold")
        self.enter = enter
        self.exit = exit
        self.hold_frames = max(1, hold_frames)
        self.per_gesture = per_gesture or {}
        self._states: Dict[str, _HandState] = {}

    def _thresholds(self, name: str) -> Tuple[float, float]:
        o = self.per_gesture.get(name, {})
        return o.get("enter", self.enter), o.get("exit", self.exit)

    def active_gesture(self, label: str) -> Optional[str]:
        st = self._states.get(label)
        return st.active if st else None

    def update(self, label: str, scores: Dict[str, float], t: float) -> List[Tuple[str, dict]]:
        """Advance one hand's state machine by one frame.

        Args:
            label: hand label ("Left"/"Right").
            scores: full gesture->confidence dict from GestureClassifier.scores().
            t: pipeline timestamp in seconds (recorded time during playback).

        Returns:
            List of (topic, payload) events to publish: "gesture.start" /
            "gesture.end". Usually empty.
        """
        st = self._states.setdefault(label, _HandState())
        events: List[Tuple[str, dict]] = []

        # --- ACTIVE: sticky until the active gesture's own score < exit ----
        if st.active is not None:
            _, exit_thr = self._thresholds(st.active)
            if scores.get(st.active, 0.0) >= exit_thr:
                st.candidate, st.count = None, 0  # nothing competes while active
                return events
            events.append(
                (
                    "gesture.end",
                    {
                        "hand": label,
                        "name": st.active,
                        "t": round(t, 4),
                        "duration": round(t - st.active_since, 4),
                    },
                )
            )
            st.active = None
            # fall through: entry logic may begin a new candidate this frame

        # --- IDLE / PENDING entry logic ------------------------------------
        best_name, best_score = max(scores.items(), key=lambda kv: kv[1])
        best_enter, _ = self._thresholds(best_name)

        if best_score >= best_enter:
            if st.candidate == best_name:
                st.count += 1
            else:
                st.candidate, st.count = best_name, 1
        elif st.candidate is not None:
            # Candidate no longer the confident winner: hysteresis grace zone.
            _, cand_exit = self._thresholds(st.candidate)
            if scores.get(st.candidate, 0.0) >= cand_exit:
                st.count += 1  # sagged but above exit: countdown continues
            else:
                st.candidate, st.count = None, 0  # below exit: abort, no event

        if st.candidate is not None and st.count >= self.hold_frames:
            st.active = st.candidate
            st.active_since = t
            events.append(
                (
                    "gesture.start",
                    {
                        "hand": label,
                        "name": st.active,
                        "score": round(scores.get(st.active, 0.0), 4),
                        "t": round(t, 4),
                    },
                )
            )
            st.candidate, st.count = None, 0
        return events

    def prune(self, present_labels, t: float) -> List[Tuple[str, dict]]:
        """Drop state for hands that left the frame, ending any active gesture."""
        events: List[Tuple[str, dict]] = []
        for label in list(self._states):
            if label in present_labels:
                continue
            st = self._states.pop(label)
            if st.active is not None:
                events.append(
                    (
                        "gesture.end",
                        {
                            "hand": label,
                            "name": st.active,
                            "t": round(t, 4),
                            "duration": round(t - st.active_since, 4),
                        },
                    )
                )
        return events
