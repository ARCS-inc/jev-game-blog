"""11 context ablation: give the model measured game state as context, every on/off combination.

Three parts, each in or out (8 conditions, "none" = 08's image-only setting):
- bird: the bird's height, vertical speed and acceleration, tracked from the red blob across frames
- posts: the next post's left/right edges and its gap's top, bottom and middle
- physics: the game's constants (gravity, a flap's effect on speed, max fall speed, ground)
Both models answer "should it flap now?" with their defaults (d1: 1 frame, P >= 0.5; Jev: 4 frames,
P >= 0.6). Jev needs TYPESAFE_API_KEY: `uv run --env-file .env python scripts/11_context_ablation.py`.
"""

from itertools import combinations
from pathlib import Path

from jev_flap.evaluate import frame_question
from jev_flap.game import play
from jev_flap.model import D1Model, JevModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / Path(__file__).stem

MODELS = {"d1": D1Model, "jev": JevModel}
PARTS = ["bird", "posts", "physics"]
CONDITIONS = [frozenset(c) for n in range(len(PARTS) + 1) for c in combinations(PARTS, n)]
PROMPT = ROOT / "prompts" / "frames" / "should_flap.json"  # no context
PROMPT_CONTEXT = ROOT / "prompts" / "frames" / "should_flap_context.json"  # same, plus "The state holds..."
FLAP_OPTION = "flap"
EPISODES = 3
MAX_TURNS = 500  # 40 s of game time; caps Jev's API spend (~4k input tokens per turn)

if __name__ == "__main__":
    for tag, cls in MODELS.items():
        model = cls()
        for parts in CONDITIONS:
            name = "+".join(p for p in PARTS if p in parts) or "none"
            question = frame_question(PROMPT_CONTEXT if parts else PROMPT, model.history)
            play(model, question, FLAP_OPTION, out_dir=OUT / tag / name, episodes=EPISODES,
                 max_turns=MAX_TURNS, context_parts=parts)
