"""05 eval frames (d1-3B): does d1-3B get more from the last 1-4 frames? Offline AUCs on frames from 02 and 03.

Every sample has 3 earlier frames, so each frame count is scored on the same samples.
"""

from pathlib import Path

from jev_flap.evaluate import frame_question, samples, save, score
from jev_flap.model import load_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / Path(__file__).stem

MODEL = "local:LiquidAI/d1-3B"
GIFS = sorted((ROOT / "outputs").glob("02_threshold_sweep/*/ep*.gif")) + sorted((ROOT / "outputs").glob("03_above_below/*/ep*.gif"))
FRAMES = [1, 2, 3, 4]
# prompts/frames/<name>.json -> (option scored, label it should predict)
QUESTIONS = {"position": ("below", "below_gap"), "motion": ("down", "falling"), "should_flap": ("flap", "flap_now")}

if __name__ == "__main__":
    seqs, labels = samples(GIFS, history=max(FRAMES))
    print(f"{len(seqs)} samples from {len(GIFS)} GIFs")
    model = load_model(MODEL)
    results = {}
    for n in FRAMES:
        for name, (option, key) in QUESTIONS.items():
            q = frame_question(ROOT / "prompts" / "frames" / f"{name}.json", n)
            r = results[f"{name}@{n}"] = score(model, [s[-n:] for s in seqs], labels, q, option, key)
            print(f"frames={n} {name:11s} AUC={r['auc']:.3f} best acc={r['best_acc']:.3f} (P>={r['best_threshold']:.3f})", flush=True)
    save(OUT / "results.json", labels, results)
