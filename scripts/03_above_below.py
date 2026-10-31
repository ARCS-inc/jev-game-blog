"""03 above / below: ask where the bird is (above or below the gap's middle) and flap when it is below.

0.41 is the most accurate threshold found offline by 04_eval_questions.
"""

from pathlib import Path

from jev_flap.game import play
from jev_flap.model import load_model, load_questions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / Path(__file__).stem

MODEL = "local:LiquidAI/d1-3B"
PROMPT = ROOT / "prompts" / "above_below.json"
FLAP_OPTION = "below"
THRESHOLDS = [0.50, 0.41]
EPISODES = 5

if __name__ == "__main__":
    model = load_model(MODEL)
    (question,) = load_questions(PROMPT).values()
    for t in THRESHOLDS:
        play(model, question, FLAP_OPTION, t, out_dir=OUT / f"threshold-{t:.2f}", episodes=EPISODES)
