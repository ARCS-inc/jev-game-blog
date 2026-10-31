"""04 eval questions: single-frame AUC of candidate questions on frames from 01 and 02.

Labels come from pixels (jev_flap.evaluate.label): `below_gap` for the gap questions, and
`lower_half` for bird_half, a perception check that ignores the posts.
"""

from pathlib import Path

from jev_flap.evaluate import samples, save, score
from jev_flap.model import load_model, load_questions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / Path(__file__).stem

MODEL = "local:LiquidAI/d1-3B"
GIFS = sorted((ROOT / "outputs").glob("02_threshold_sweep/*/ep*.gif")) + sorted((ROOT / "outputs").glob("01_first_attempt/realtime-clock/ep*.gif"))
# prompt file -> (option scored as P(flap), label it should predict)
CANDIDATES = {
    "flap_wait.json": ("flap", "below_gap"),
    "above_below.json": ("below", "below_gap"),
    "lower_noul.json": (None, "below_gap"),
    "bird_half.json": ("lower", "lower_half"),
}

if __name__ == "__main__":
    seqs, labels = samples(GIFS, history=1)
    print(f"{len(seqs)} labelled frames from {len(GIFS)} GIFs")
    model = load_model(MODEL)
    results = {}
    for file, (option, key) in CANDIDATES.items():
        (q,) = load_questions(ROOT / "prompts" / file).values()
        r = results[Path(file).stem] = score(model, seqs, labels, q, option, key)
        print(f"{Path(file).stem:12s} label={key:10s} AUC={r['auc']:.3f} best acc={r['best_acc']:.3f} (P>={r['best_threshold']:.3f})")
    save(OUT / "results.json", labels, results)
