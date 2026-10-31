"""06 eval frames, Jev vs d1-3B: the 05 evaluation on 1, 2 and 4 frames, only episode 0 of each run, for both models.

Jev runs through the TypeSafe API (TYPESAFE_API_KEY from .env): `uv run --env-file .env python scripts/06_...py`.
"""

from pathlib import Path

from jev_flap.evaluate import frame_question, samples, save, score
from jev_flap.model import load_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / Path(__file__).stem

# name -> (model spec, parallel calls)
MODELS = {"jev": ("api:jev-latest", 8), "d1": ("local:LiquidAI/d1-3B", 1)}
GIFS = sorted((ROOT / "outputs").glob("02_threshold_sweep/*/ep0.gif")) + sorted((ROOT / "outputs").glob("03_above_below/*/ep0.gif"))
FRAMES = [1, 2, 4]
HISTORY = 4  # same samples as 05: every sample has 3 earlier frames
QUESTIONS = {"position": ("below", "below_gap"), "motion": ("down", "falling"), "should_flap": ("flap", "flap_now")}

if __name__ == "__main__":
    seqs, labels = samples(GIFS, history=HISTORY)
    print(f"{len(seqs)} samples from {len(GIFS)} GIFs")
    for tag, (spec, workers) in MODELS.items():
        model = load_model(spec)
        results = {}
        for n in FRAMES:
            for name, (option, key) in QUESTIONS.items():
                q = frame_question(ROOT / "prompts" / "frames" / f"{name}.json", n)
                r = results[f"{name}@{n}"] = score(model, [s[-n:] for s in seqs], labels, q, option, key, workers)
                print(f"{tag} frames={n} {name:11s} AUC={r['auc']:.3f} best acc={r['best_acc']:.3f} "
                      f"(P>={r['best_threshold']:.3f}) p50 {r['p50_latency_ms']:.0f}ms {r['input_tokens']} tokens", flush=True)
        save(OUT / f"{tag}.json", labels, results)
