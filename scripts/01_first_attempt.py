"""01 first attempt: d1-3B is asked what to do (flap / wait) and flaps when P(flap) >= 0.5.

outputs/01_first_attempt/realtime-clock/ is the original run, from an early version whose Playwright
clock still ran in real time (about 255 ms of game time per decision, not 80 ms). This script
reproduces the attempt with the frozen clock, into outputs/01_first_attempt/.
"""

from pathlib import Path

from jev_flap.game import play
from jev_flap.model import load_model, load_questions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / Path(__file__).stem

MODEL = "local:LiquidAI/d1-3B"
PROMPT = ROOT / "prompts" / "flap_wait.json"
FLAP_OPTION = "flap"
THRESHOLD = 0.5
EPISODES = 3

if __name__ == "__main__":
    model = load_model(MODEL)
    (question,) = load_questions(PROMPT).values()
    play(model, question, FLAP_OPTION, THRESHOLD, out_dir=OUT, episodes=EPISODES)
