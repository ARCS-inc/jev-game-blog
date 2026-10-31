"""13 Jev gap threshold sweep: 12 plus the next post and its gap (bird+posts+physics context).

In 12, without the gap's position Jev flapped whenever the bird started falling, even near the top,
and hit the upper post. Here it also gets the next post's edges and gap (top, bottom, middle).
Same setting otherwise (4 frames, "should it flap now?"), thresholds around 12's best (0.45).
`uv run --env-file .env python scripts/13_jev_gap_threshold.py`
"""

from pathlib import Path

from jev_flap.evaluate import frame_question
from jev_flap.game import play
from jev_flap.model import JevModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / Path(__file__).stem

PROMPT = ROOT / "prompts" / "frames" / "should_flap_context.json"
FLAP_OPTION = "flap"
CONTEXT = frozenset({"bird", "posts", "physics"})
THRESHOLDS = [0.4, 0.45, 0.5, 0.55]
EPISODES = 3
MAX_TURNS = 500  # 40 s of game time; caps the API spend

if __name__ == "__main__":
    model = JevModel()
    question = frame_question(PROMPT, model.history)
    for threshold in THRESHOLDS:
        play(model, question, FLAP_OPTION, threshold, out_dir=OUT / f"thr{threshold}", episodes=EPISODES,
             max_turns=MAX_TURNS, context_parts=CONTEXT)
