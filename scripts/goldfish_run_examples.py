"""金魚すくいの画像を Jev に見せて、「金魚はお椀の中か」を確率で答えさせる。

    uv run python scripts/goldfish_make_images.py
    uv run --env-file .env python scripts/goldfish_run_examples.py            # jev-latest
    uv run --env-file .env python scripts/goldfish_run_examples.py jev-preview

結果は outputs/goldfish_run_examples/<model>/ に出る。
- results.json  : 画像ごとの答え（確率）と推論時間
- grid.png      : 画像に P(お椀の中) を重ねた一覧（ブログ用）
- series.png    : 金魚が外から中へ移動したときの P(お椀の中) の推移（ブログ用）
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from typesafe_sdk import Image as AssetImage
from typesafe_sdk import TypeSafeClient

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "assets" / "goldfish"
OUT = ROOT / "outputs" / Path(__file__).stem

QUESTIONS = {
    "inside": {
        "type": "noul",
        "instructions": "This is a top-down illustration of a goldfish scooping game. "
                        "Are all the goldfish entirely inside the white bowl?",
    },
    "where": {
        "type": "choice",
        "instructions": "This is a top-down illustration of a goldfish scooping game. Where is the goldfish?",
        "criteria": {
            "inside": "inside the bowl",
            "rim": "on the rim of the bowl",
            "outside": "outside the bowl, in the pool",
        },
    },
}


def ask(client: TypeSafeClient, path: Path) -> dict:
    img = AssetImage.from_pil(Image.open(path).convert("RGB"), format="png")
    t = time.perf_counter()
    res = client.system_one(state=img, questions=QUESTIONS)  # type: ignore[arg-type]
    ms = (time.perf_counter() - t) * 1000
    answers = {k: a.model_dump(mode="json") for k, a in res.answers.items()}
    return {"image": path.name, "model": res.model, "latency_ms": round(ms),
            "input_tokens": res.usage.input_tokens if res.usage else None, "answers": answers}


def p_yes(row: dict) -> float:
    """noul の答えから P(はい) を取り出す。"""
    return float(row["answers"]["inside"]["noul"])


def font(size: int) -> ImageFont.ImageFont:
    for name in ("NotoSansCJK-Bold.ttc", "NotoSansCJKjp-Bold.otf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def grid(rows: list[dict], out: Path, cols: int = 4, w: int = 320, h: int = 240) -> None:
    n = len(rows)
    sheet = Image.new("RGB", (cols * w, -(-n // cols) * h), "white")
    f = font(26)
    for i, row in enumerate(rows):
        im = Image.open(IMAGES / row["image"]).convert("RGB").resize((w, h))
        d = ImageDraw.Draw(im, "RGBA")
        p = p_yes(row)
        d.rectangle([0, h - 44, w, h], fill=(0, 0, 0, 170))
        d.text((10, h - 40), f"P(in bowl) = {p:.2f}", fill="white", font=f)
        bar = int((w - 20) * p)
        d.rectangle([10, h - 6, 10 + bar, h - 2], fill=(255, 200, 0, 255))
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(out)


def series_plot(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted((r for r in rows if r["image"].startswith("series_")),
                  key=lambda r: int(r["image"].split("_")[1].split(".")[0]))
    if not rows:
        return
    xs = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.plot(xs, [p_yes(r) for r in rows], marker="o", color="#ff6a2b", lw=2.5)
    ax.axhline(0.5, color="#999", lw=1, ls="--")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(xs)
    ax.set_xlabel("frame (fish moves from the pool into the bowl →)")
    ax.set_ylabel("P(inside the bowl)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "jev-latest"
    out = OUT / model
    out.mkdir(parents=True, exist_ok=True)
    client = TypeSafeClient(model=model, headers={"TypeSafe-Danger-Zone-Images": "true"}, timeout=30.0)
    paths = sorted(IMAGES.glob("series_*.png")) + sorted(IMAGES.glob("extra_*.png"))
    rows = []
    for path in paths:
        row = ask(client, path)
        rows.append(row)
        where = row["answers"]["where"]
        print(f"{row['image']:28s} P(in)={p_yes(row):.2f}  where={where.get('choice')}  {row['latency_ms']}ms")
    client.close()
    (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    grid(rows, out / "grid.png")
    series_plot(rows, out / "series.png")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
