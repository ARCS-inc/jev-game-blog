"""10 d1 position on 2 frames every 20 ms: 09 with the last 2 frames instead of 1.

05 found d1-3B's position AUC best on 2 frames (0.84 vs 0.77 on 1), though 06 (episode 0 only)
didn't confirm it. The question is 03's above / below, worded for several images. Everything
else matches 09, so the two compare 1 vs 2 frames.
"""

from pathlib import Path

from jev_flap.evaluate import frame_question
from jev_flap.game import play
from jev_flap.model import D1Model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / Path(__file__).stem

HISTORY = 2
PROMPT = ROOT / "prompts" / "frames" / "position.json"
FLAP_OPTION = "below"
THRESHOLDS = [0.41, 0.50]
STEP_MS = 20
EPISODES = 5

if __name__ == "__main__":
    model = D1Model(history=HISTORY)
    question = frame_question(PROMPT, HISTORY)
    for t in THRESHOLDS:
        play(model, question, FLAP_OPTION, t, out_dir=OUT / f"threshold-{t:.2f}", episodes=EPISODES, step_ms=STEP_MS)
