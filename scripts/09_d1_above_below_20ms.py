"""09 d1 above / below every 20 ms: 03 with a decision every 20 ms of game time instead of 80 ms.

The clock is frozen, so latency doesn't matter; this looks for d1-3B's ceiling when it can
react about once per rendered frame (60 fps = 16.7 ms).
"""

from pathlib import Path

from jev_flap.game import play
from jev_flap.model import D1Model, load_questions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / Path(__file__).stem

PROMPT = ROOT / "prompts" / "above_below.json"
FLAP_OPTION = "below"
THRESHOLDS = [0.41, 0.50]
STEP_MS = 20
EPISODES = 5

if __name__ == "__main__":
    model = D1Model()
    (question,) = load_questions(PROMPT).values()
    for t in THRESHOLDS:
        play(model, question, FLAP_OPTION, t, out_dir=OUT / f"threshold-{t:.2f}", episodes=EPISODES, step_ms=STEP_MS)
