"""07 Jev, should it flap?: Jev sees the last 4 frames and is asked directly whether to flap now.

The question and threshold are the best offline setting from 06 (should_flap on 4 frames, AUC 0.86;
P(flap) >= 0.78 was the most accurate threshold). Needs TYPESAFE_API_KEY:
`uv run --env-file .env python scripts/07_jev_should_flap.py`.
"""

from pathlib import Path

from jev_flap.evaluate import frame_question
from jev_flap.game import play
from jev_flap.model import load_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / Path(__file__).stem

MODEL = "api:jev-latest"
HISTORY = 4
PROMPT = ROOT / "prompts" / "frames" / "should_flap.json"
FLAP_OPTION = "flap"
THRESHOLD = 0.78
EPISODES = 3
MAX_TURNS = 1500  # caps the API spend of one episode (~4k input tokens per turn)

if __name__ == "__main__":
    model = load_model(MODEL)
    question = frame_question(PROMPT, HISTORY)
    play(model, question, FLAP_OPTION, THRESHOLD, out_dir=OUT, episodes=EPISODES,
         history=HISTORY, max_turns=MAX_TURNS)
