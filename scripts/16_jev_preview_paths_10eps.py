"""16 drawn paths, more rounds: 15's best setting (0.3 s paths, threshold 0.5) over 10 rounds.

15 scored 0 / 21 / 56 at threshold 0.5. The model's answers matched the exact ones on the frames
checked; how far the bird gets depends on whether its bobbing lines up with each post, so three
rounds say little. Same model, questions, rule and paths as 15.
`uv run --env-file .env python scripts/16_jev_preview_paths_10eps.py`
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
THRESHOLD = 0.5
EPISODES = 10
MAX_TURNS = 1500  # 120 s of game time, about 58 posts


def policy(answers):
    return answers["wait_crash"]["noul"] * (1 - answers["flap_crash"]["noul"])


if __name__ == "__main__":
    model = JevModel(MODEL, history=1)
    play(model, QUESTIONS, None, THRESHOLD, out_dir=OUT, episodes=EPISODES, max_turns=MAX_TURNS,
         paths=True, policy=policy)
