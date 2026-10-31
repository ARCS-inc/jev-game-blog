"""金魚すくいのテスト画像を作る。

金魚が「お椀の外（水槽）」から「お椀の中」へ少しずつ移動する9枚と、おまけの数枚を
SVG で描き、Chromium で PNG にする。すべて自作の図なので、ライセンスの心配はない。

    uv run python scripts/goldfish_make_images.py   # assets/goldfish/*.png ができる
"""

from __future__ import annotations

import math
from pathlib import Path

from playwright.sync_api import sync_playwright

W, H = 640, 480
BOWL = (400, 250, 140)  # お椀の中心 x, y と、内側の半径
OUT = Path(__file__).resolve().parents[1] / "assets" / "goldfish"


def fish(x: float, y: float, angle: float = 0, color: str = "#ff6a2b", scale: float = 1.0,
         kind: str = "wakin") -> str:
    """上から見た金魚。(x, y) は体の中心、angle は頭の向き（度、0 = 右）。"""
    if kind == "demekin":  # 黒い出目金
        body, fin, eye = "#2b2b33", "#44444f", "#f5f5f5"
        eyes = ('<circle cx="22" cy="-13" r="8" fill="{e}"/><circle cx="22" cy="13" r="8" fill="{e}"/>'
                '<circle cx="24" cy="-13" r="4" fill="#111"/><circle cx="24" cy="13" r="4" fill="#111"/>').format(e=eye)
    else:
        body, fin = color, "#ffb08a"
        eyes = ('<circle cx="24" cy="-8" r="4" fill="#fff"/><circle cx="24" cy="8" r="4" fill="#fff"/>'
                '<circle cx="25" cy="-8" r="2" fill="#111"/><circle cx="25" cy="8" r="2" fill="#111"/>')
    return f"""
    <g transform="translate({x},{y}) rotate({angle}) scale({scale})">
      <path d="M-28,0 C-50,-30 -78,-34 -86,-22 C-72,-10 -72,10 -86,22 C-78,34 -50,30 -28,0 Z"
            fill="{fin}" opacity="0.9"/>
      <path d="M-10,-14 C-4,-32 10,-32 8,-16 Z" fill="{fin}"/>
      <path d="M-10,14 C-4,32 10,32 8,16 Z" fill="{fin}"/>
      <ellipse cx="0" cy="0" rx="36" ry="20" fill="{body}"/>
      <path d="M-20,-6 Q0,-12 20,-6" stroke="#fff" stroke-opacity="0.35" stroke-width="3" fill="none"/>
      {eyes}
    </g>"""


def scene(fishes: list[str], bowl: bool = True, poi: bool = True) -> str:
    bx, by, br = BOWL
    ripples = "".join(
        f'<ellipse cx="{(i * 137) % W}" cy="{(i * 89) % H}" rx="{30 + i % 3 * 12}" ry="6" '
        f'fill="none" stroke="#ffffff" stroke-opacity="0.25" stroke-width="2"/>' for i in range(14))
    bowl_svg = f"""
      <circle cx="{bx + 8}" cy="{by + 10}" r="{br + 18}" fill="#000" opacity="0.15"/>
      <circle cx="{bx}" cy="{by}" r="{br + 18}" fill="#fdfdfd" stroke="#d9d9d9" stroke-width="3"/>
      <circle cx="{bx}" cy="{by}" r="{br}" fill="#bfe6ff"/>
      <circle cx="{bx}" cy="{by}" r="{br}" fill="none" stroke="#9fd0f0" stroke-width="4"/>
      <circle cx="{bx - 40}" cy="{by - 50}" r="16" fill="#fff" opacity="0.5"/>
    """ if bowl else ""
    poi_svg = """
      <g transform="translate(90,390) rotate(-30)">
        <rect x="-8" y="0" width="16" height="110" rx="6" fill="#e83e5a"/>
        <circle cx="0" cy="-40" r="46" fill="#ffffff" fill-opacity="0.75" stroke="#e83e5a" stroke-width="8"/>
      </g>""" if poi else ""
    # 外の金魚はお椀の下、中の金魚はお椀の上に描く（呼び出し側で順序を決める）
    under, over = fishes[0], "".join(fishes[1:])
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
      <rect width="{W}" height="{H}" fill="#2f8fd8"/>
      {ripples}
      {under}
      {bowl_svg}
      {over}
      {poi_svg}
    </svg>"""


def series() -> dict[str, str]:
    """金魚が左の水槽からお椀の中心へ向かって9段階で移動する。"""
    bx, by, br = BOWL
    out = {}
    start, end = 90, bx  # 金魚の中心 x
    for i in range(9):
        x = start + (end - start) * i / 8
        y = by + 20 * math.sin(i / 8 * math.pi)  # 少しだけ弧を描く
        inside = abs(x - bx) < br  # 中心がお椀の内側か
        f = fish(x, y, angle=0)
        fishes = ["", f] if inside else [f]
        out[f"series_{i}"] = scene(fishes)
    return out


def extras() -> dict[str, str]:
    bx, by, _ = BOWL
    return {
        "extra_empty_bowl": scene([fish(150, 120, 30), fish(180, 380, -150)]),
        "extra_demekin_in_bowl": scene(["", fish(bx + 10, by - 10, 120, kind="demekin")]),
        "extra_two_in_bowl": scene(["", fish(bx - 50, by + 30, 200), fish(bx + 50, by - 30, 20, color="#ff3b3b")]),
        "extra_under_bowl_edge": scene([fish(bx - 150, by + 40, 10)]),  # お椀の縁の下に半分隠れている
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    scenes = {**series(), **extras()}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": W, "height": H})
        for name, svg in scenes.items():
            page.set_content(f'<html><body style="margin:0">{svg}</body></html>')
            page.screenshot(path=str(OUT / f"{name}.png"))
            (OUT / f"{name}.svg").write_text(svg)
        browser.close()
    print(f"{len(scenes)} images -> {OUT}")


if __name__ == "__main__":
    main()
