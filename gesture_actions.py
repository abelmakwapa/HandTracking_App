"""
gesture_actions.py
==================

GestureActionMapper: user-editable bindings from gesture/button events to
actions (keystrokes or shell commands), loaded from a JSON or YAML file.

Config schema (see gesture_config.json for a working sample):

    {
      "stabilizer": {                      # optional; feeds GestureStabilizer
        "enter": 0.75, "exit": 0.55, "hold_frames": 4,
        "per_gesture": {"ok": {"enter": 0.8, "exit": 0.6}}
      },
      "bindings": [
        {"on": "gesture:thumbs_up",        # event key, see below
         "hand": "any",                    # "Left" | "Right" | "any"
         "action": {"type": "key", "keys": ["space"]},
         "cooldown": 1.0},                 # seconds between firings
        {"on": "dynamic:swipe:left",
         "action": {"type": "key", "keys": ["left"]}}
      ],
      "buttons": [ ... ]                   # consumed by virtual_buttons.py
    }

Event keys are colon-joined:
    gesture:<name>            stabilized static gesture START (debounced —
                              the hold-time/hysteresis machine has already
                              filtered out single-frame misdetections)
    dynamic:<name>:<detail>   motion gesture, e.g. dynamic:swipe:left
    interaction:<name>:<detail>  two-hand event, e.g. interaction:touch:palms close
    button:<name>             virtual button press

A binding matches an event if its key equals the event key OR is a prefix at
a ':' boundary — so "dynamic:swipe" matches every swipe direction.

Action types:
    {"type": "key", "keys": ["ctrl", "t"]}   -> pyautogui.hotkey("ctrl", "t")
    {"type": "shell", "command": "echo hi"}  -> subprocess.Popen (non-blocking)

Shell commands run with your user's privileges — treat the config file with
the same trust you'd give a shell script.
"""

import subprocess
from typing import List, Optional

# Topic -> function that builds the colon-joined event key from the payload.
_KEY_BUILDERS = {
    "gesture.start": lambda p: f"gesture:{p['name']}",
    "dynamic": lambda p: f"dynamic:{p['name']}:{p['detail']}",
    "interaction": lambda p: f"interaction:{p['name']}:{p['detail']}",
    "button.press": lambda p: f"button:{p['name']}",
}


def load_config(path: str) -> dict:
    """Load a JSON (stdlib) or YAML (requires pyyaml) gesture config file."""
    import json
    import os

    if not os.path.exists(path):
        raise SystemExit(f"[error] Gesture config not found: {path!r}")
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError:
            raise SystemExit(
                "[error] YAML config requires pyyaml (pip install pyyaml), "
                "or use a .json config instead."
            )
        return yaml.safe_load(text) or {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[error] Invalid JSON in {path!r}: {exc}")


class _Binding:
    def __init__(self, cfg: dict):
        try:
            self.key: str = cfg["on"]
            self.action: dict = cfg["action"]
        except KeyError as exc:
            raise SystemExit(f"[error] Config binding missing field {exc}: {cfg}")
        self.hand: str = cfg.get("hand", "any")
        self.cooldown: float = float(cfg.get("cooldown", 0.5))
        self.last_fired: float = -1e9
        atype = self.action.get("type")
        if atype not in ("key", "shell"):
            raise SystemExit(
                f"[error] Binding {self.key!r}: action type must be 'key' or "
                f"'shell', got {atype!r}."
            )

    def matches(self, event_key: str, hand: Optional[str]) -> bool:
        if not (event_key == self.key or event_key.startswith(self.key + ":")):
            return False
        return self.hand == "any" or hand is None or hand == self.hand


class GestureActionMapper:
    """Subscribes to gesture/button events and fires the configured actions.

    Decoupling: constructed only when a config with bindings is supplied; the
    pipeline publishes the same events regardless.
    """

    def __init__(self, config: dict, bus):
        self._bindings: List[_Binding] = [_Binding(c) for c in config.get("bindings", [])]
        self._pyautogui = None
        for topic in _KEY_BUILDERS:
            bus.subscribe(topic, self._make_handler(topic))
        print(f"[info] Action mapper: {len(self._bindings)} binding(s) loaded.")

    def _make_handler(self, topic: str):
        build_key = _KEY_BUILDERS[topic]

        def handler(payload: dict) -> None:
            event_key = build_key(payload)
            hand = payload.get("hand")
            t = payload.get("t", 0.0)
            for b in self._bindings:
                if not b.matches(event_key, hand):
                    continue
                if (t - b.last_fired) < b.cooldown:
                    continue
                b.last_fired = t
                self._execute(b, event_key)

        return handler

    def _execute(self, binding: _Binding, event_key: str) -> None:
        action = binding.action
        if action["type"] == "key":
            keys = action.get("keys", [])
            if isinstance(keys, str):
                keys = [keys]
            try:
                if self._pyautogui is None:
                    import pyautogui
                    pyautogui.PAUSE = 0.0
                    self._pyautogui = pyautogui
                self._pyautogui.hotkey(*keys)
                print(f"[action] {event_key} -> key {'+'.join(keys)}")
            except Exception as exc:
                print(f"[warn] Key action for {event_key!r} failed: {exc}")
        else:  # shell
            cmd = action.get("command", "")
            try:
                # Popen (not run): fire-and-forget so a slow command can't
                # stall the video loop.
                subprocess.Popen(
                    cmd, shell=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                print(f"[action] {event_key} -> shell: {cmd}")
            except Exception as exc:
                print(f"[warn] Shell action for {event_key!r} failed: {exc}")
