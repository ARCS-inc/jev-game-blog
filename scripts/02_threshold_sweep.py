"""02 threshold sweep: the first attempt's question with flap thresholds between d1-3B's few P(flap) values.

P(flap) only took 0.531 / 0.562 / 0.593 / 0.622 (bf16 steps), so each threshold sits between two of them.
"""

from pathlib import Path

from jev_flap.game import play
from jev_flap.model import load_model, load_questions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / Path(__file__).stem

MODEL = "local:LiquidAI/d1-3B"
PROMPT = ROOT / "prompts" / "flap_wait.json"
FLAP_OPTION = "flap"
THRESHOLDS = [0.50, 0.55, 0.57, 0.60]
EPISODES = 3

if __name__ == "__main__":
    model = load_model(MODEL)
    (question,) = load_questions(PROMPT).values()
    for t in THRESHOLDS:
        play(model, question, FLAP_OPTION, t, out_dir=OUT / f"threshold-{t:.2f}", episodes=EPISODES)
