"""08 Jev, should it flap? (lower threshold): 07 again with JevModel's defaults, flapping from P(flap) >= 0.6.

In 07 (threshold 0.78, from offline accuracy) Jev's P(flap) sat mostly at 0.6-0.78 while the bird
had to climb, so it barely flapped and hit the lower posts. Needs TYPESAFE_API_KEY:
`uv run --env-file .env python scripts/08_jev_should_flap_default.py`.
"""

from pathlib import Path

from jev_flap.evaluate import frame_question
from jev_flap.game import play
from jev_flap.model import JevModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / Path(__file__).stem

PROMPT = ROOT / "prompts" / "frames" / "should_flap.json"
FLAP_OPTION = "flap"
EPISODES = 3
MAX_TURNS = 1500  # caps the API spend of one episode (~4k input tokens per turn)

if __name__ == "__main__":
    model = JevModel()  # threshold 0.6, last 4 frames
    question = frame_question(PROMPT, model.history)
    play(model, question, FLAP_OPTION, out_dir=OUT, episodes=EPISODES, max_turns=MAX_TURNS)
