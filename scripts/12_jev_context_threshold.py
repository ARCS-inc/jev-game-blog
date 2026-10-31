"""12 Jev context threshold sweep: 11's bird+physics context, with lower thresholds.

In 11, bird+physics separated "should flap" turns perfectly (AUC 1.0), but P(flap) on those turns
averaged only 0.55, under Jev's default threshold 0.6, so the bird rarely flapped. Same setting
(4 frames, "should it flap now?"), thresholds swept below 0.6.
`uv run --env-file .env python scripts/12_jev_context_threshold.py`
"""

from pathlib import Path

from jev_flap.evaluate import frame_question
from jev_flap.game import play
from jev_flap.model import JevModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / Path(__file__).stem

PROMPT = ROOT / "prompts" / "frames" / "should_flap_context.json"
FLAP_OPTION = "flap"
CONTEXT = frozenset({"bird", "physics"})
THRESHOLDS = [0.35, 0.4, 0.45, 0.5]
EPISODES = 3
MAX_TURNS = 500  # 40 s of game time; caps the API spend

if __name__ == "__main__":
    model = JevModel()
    question = frame_question(PROMPT, model.history)
    for threshold in THRESHOLDS:
        play(model, question, FLAP_OPTION, threshold, out_dir=OUT / f"thr{threshold}", episodes=EPISODES,
             max_turns=MAX_TURNS, context_parts=CONTEXT)
