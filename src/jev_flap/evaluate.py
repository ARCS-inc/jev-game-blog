"""Offline evaluation of questions on Flap frames saved by `game.play` (one GIF frame per turn).

Frames are labelled from their pixels: the bird is the red blob, the next post is the nearest run
of green columns to the right of the bird's left edge, and the gap is the opening between that
post's upper and lower parts. A question is scored by how well its probability separates a label
(AUC), and by its accuracy at the best threshold.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from jev_flap.game import p_flap
from jev_flap.harness import detect
from jev_flap.model import DecisionModel

MIN_MOVE_PX = 2.0


def bird_y(frame: Image.Image) -> float | None:
    bird = detect(frame)["bird"]
    return bird[1] if bird else None


def label(frame: Image.Image) -> dict | None:
    """Bird and gap of one frame, in pixels, or None when the bird or a post ahead of it isn't visible."""
    d = detect(frame)
    if not d["bird"] or not d["post"]:
        return None
    by, h = d["bird"][1], d["size"][1]
    gap_mid = (d["post"]["gap_top"] + d["post"]["gap_bottom"]) / 2
    return {"bird_y": by, "gap_mid": gap_mid, "below_gap": bool(by > gap_mid), "lower_half": bool(by > h / 2)}


def gif_frames(path: Path) -> list[Image.Image]:
    im = Image.open(path)
    out = []
    for i in range(im.n_frames):
        im.seek(i)
        out.append(im.convert("RGB"))
    return out


def samples(gifs: Sequence[Path], history: int) -> tuple[list[list[Image.Image]], list[dict]]:
    """Every labelled frame with `history - 1` earlier frames, as (frame sequences, labels).

    With `history` > 1, labels add `falling` (moved down since the previous frame) and `flap_now`
    (below the gap's middle and falling), and frames where the bird barely moved are dropped,
    since "up or down" has no answer there. Slice `seq[-n:]` to score fewer frames on the same samples.
    """
    seqs, labels = [], []
    for g in gifs:
        frames = gif_frames(g)
        for t in range(history - 1, len(frames)):
            lab = label(frames[t])
            if not lab:
                continue
            if history > 1:
                prev = bird_y(frames[t - 1])
                if prev is None or abs(lab["bird_y"] - prev) < MIN_MOVE_PX:
                    continue
                lab["falling"] = lab["bird_y"] > prev
                lab["flap_now"] = lab["below_gap"] and lab["falling"]
            seqs.append(frames[t - history + 1:t + 1])
            labels.append({"gif": str(g), "frame": t, **lab})
    return seqs, labels


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def score(model: DecisionModel, seqs: Sequence[Sequence[Image.Image]], labels: Sequence[dict],
          question: Mapping[str, Any], flap_option: str | None, key: str, workers: int = 1) -> dict:
    """Ask `question` about every sequence (all its frames) and score P(`flap_option`) against `key`."""
    ask: Callable = lambda s: (model.predict(s[0], {"q": question}) if len(s) == 1
                               else model.predict_frames(s, {"q": question}))
    with ThreadPoolExecutor(workers) as pool:
        decisions = list(pool.map(ask, seqs))
    probs = [p_flap(d.answers["q"], flap_option) for d in decisions]
    ys = [l[key] for l in labels]
    best_acc, best_t = max((sum((p >= t) == y for p, y in zip(probs, ys)) / len(ys), t)
                           for t in sorted(set(probs)))
    ms = sorted(d.latency_ms for d in decisions)
    return {"model": decisions[0].model, "label": key, "auc": auc(probs, ys), "best_acc": best_acc,
            "best_threshold": best_t, "p50_latency_ms": ms[len(ms) // 2],
            "input_tokens": sum(d.input_tokens or 0 for d in decisions), "probs": probs}


def save(path: Path, labels: Sequence[dict], results: Mapping[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"labels": list(labels), "results": dict(results)}, indent=1))


def frame_question(path: Path, n: int) -> dict:
    """A question template from prompts/frames/ for `n` frames: fills `{lead}` and `{now}` from the
    `lead.json` next to it (a one-image or an n-image intro, and where to look)."""
    lead = json.loads((path.parent / "lead.json").read_text())
    q = json.loads(path.read_text())
    q["instructions"] = q["instructions"].format(
        lead=lead["one"] if n == 1 else lead["many"].format(n=n),
        now=lead["now_one"] if n == 1 else lead["now_many"])
    return q
