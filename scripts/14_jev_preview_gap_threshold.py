"""14 Jev preview: 13's setting (bird+posts+physics context, threshold sweep) with jev-preview.

The API lists two models: jev-latest (used so far) and jev-preview ("A preview version of
`jev-latest`: should be better in most ways"). Everything else, including JevModel's defaults
(4 frames), is the same as 13.
`uv run --env-file .env python scripts/14_jev_preview_gap_threshold.py`
"""

from pathlib import Path

from jev_flap.evaluate import frame_question
from jev_flap.game import play
from jev_flap.model import JevModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / Path(__file__).stem

MODEL = "jev-preview"
PROMPT = ROOT / "prompts" / "frames" / "should_flap_context.json"
FLAP_OPTION = "flap"
CONTEXT = frozenset({"bird", "posts", "physics"})
THRESHOLDS = [0.4, 0.45, 0.5, 0.55]
EPISODES = 3
MAX_TURNS = 500  # 40 s of game time; caps the API spend

if __name__ == "__main__":
    model = JevModel(MODEL)
    question = frame_question(PROMPT, model.history)
    for threshold in THRESHOLDS:
        play(model, question, FLAP_OPTION, threshold, out_dir=OUT / f"thr{threshold}", episodes=EPISODES,
             max_turns=MAX_TURNS, context_parts=CONTEXT)
