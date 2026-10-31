"""15 drawn paths: draw where the bird goes if it waits or flaps, and ask whether each one crashes.

In 13-14 Jev barely used the numbers it was given (the gap's position hardly moved P(flap)). Here
the harness turns them into a picture instead: on the current frame it draws the bird's next 0.3 s
if it doesn't flap (blue) and if it flaps now (orange), as wide as its hitbox and shifted at the
posts' scroll speed, so a band touching green or the ground means a crash (harness.draw_paths).
Jev answers two yes/no questions about the one frame, and the bird flaps when
P(blue crashes) x P(orange is clear) >= the threshold.

In a Python copy of the game, this rule with exact answers scored 41 on average over 8 rounds
with 0.3 s paths (0.25 s: 7, 0.5 s: 9). jev-preview, the steadier of the two Jev models in 14.
`uv run --env-file .env python scripts/15_jev_preview_paths.py`
"""

import json
from pathlib import Path

from jev_flap.game import play
from jev_flap.model import JevModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / Path(__file__).stem

MODEL = "jev-preview"
QUESTIONS = {name: json.loads((ROOT / "prompts" / "paths" / f"{name}.json").read_text())
             for name in ("wait_crash", "flap_crash")}
THRESHOLDS = [0.5, 0.3]
EPISODES = 3
MAX_TURNS = 1500  # 120 s of game time, about 58 posts


def policy(answers):
    return answers["wait_crash"]["noul"] * (1 - answers["flap_crash"]["noul"])


if __name__ == "__main__":
    model = JevModel(MODEL, history=1)
    for threshold in THRESHOLDS:
        play(model, QUESTIONS, None, threshold, out_dir=OUT / f"thr{threshold}", episodes=EPISODES,
             max_turns=MAX_TURNS, paths=True, policy=policy)
