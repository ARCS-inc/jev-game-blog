"""金魚すくいの結果から4枚を選び、画像の下に Jev の答え（確率）を描いてブログ用に保存する。

    uv run python scripts/goldfish_blog_figures.py            # jev-latest
    uv run python scripts/goldfish_blog_figures.py jev-preview

先に scripts/goldfish_run_examples.py を実行しておくこと。
assets/goldfish-answer-<画像名>.png ができる。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "assets" / "goldfish"
RESULTS = ROOT / "outputs" / "goldfish_run_examples"

# 外 → 境目で迷う → 中 → 色違いの金魚
PICKS = ["series_0.png", "series_5.png", "series_7.png", "extra_demekin_in_bowl.png"]
WHERE_LABELS = {"inside": "お椀の中", "rim": "お椀の縁", "outside": "お椀の外"}

W, H = 640, 480
PANEL = 244
ORANGE, GRAY, TEXT, SUB = (255, 106, 43), (215, 215, 215), (34, 34, 34), (110, 110, 110)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (["ヒラギノ角ゴシック W6.ttc", "NotoSansCJK-Bold.ttc"] if bold
             else ["ヒラギノ角ゴシック W3.ttc", "NotoSansCJK-Regular.ttc"])
    for name in names:
        for path in (Path("/System/Library/Fonts") / name, Path(name)):
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def bar(d: ImageDraw.ImageDraw, y: int, label: str, p: float, highlight: bool) -> None:
    x0, x1 = 150, W - 90
    d.text((24, y), label, fill=TEXT, font=font(20))
    d.rounded_rectangle([x0, y + 4, x1, y + 24], radius=4, fill=(242, 242, 242))
    if p > 0.005:
        d.rounded_rectangle([x0, y + 4, x0 + max(8, int((x1 - x0) * p)), y + 24], radius=4,
                            fill=ORANGE if highlight else GRAY)
    d.text((x1 + 12, y), f"{p:.2f}", fill=TEXT if highlight else SUB, font=font(20, bold=highlight))


def figure(row: dict, out: Path) -> None:
    im = Image.new("RGB", (W, H + PANEL), "white")
    im.paste(Image.open(IMAGES / row["image"]).convert("RGB").resize((W, H)), (0, 0))
    d = ImageDraw.Draw(im)

    y = H + 16
    d.text((24, y), "すべての金魚がお椀の中に入っている？", fill=TEXT, font=font(20, bold=True))
    p_in = float(row["answers"]["inside"]["noul"])
    bar(d, y + 34, "はい", p_in, highlight=True)

    y += 84
    where = row["answers"]["where"]
    d.text((24, y), "金魚はどこにいる？", fill=TEXT, font=font(20, bold=True))
    for i, (key, label) in enumerate(WHERE_LABELS.items()):
        bar(d, y + 34 + i * 30, label, where["probabilities"][key], highlight=key == where["choice"])
    im.save(out)


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "jev-latest"
    rows = {r["image"]: r for r in json.loads((RESULTS / model / "results.json").read_text())}
    for name in PICKS:
        out = ROOT / "assets" / f"goldfish-answer-{Path(name).stem}.png"
        figure(rows[name], out)
        print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
