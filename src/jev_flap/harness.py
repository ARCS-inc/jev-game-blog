"""Game state measured from Flap screenshots, to give a model as context (System One `state`).

Units follow the game (opengames/flap/logic.js): the board is 1 x 1, y grows downward, speeds are
per second. Everything here is read from pixels, except `PHYSICS`, the game's constants.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

POST_COLORS = np.array([(111, 143, 114), (76, 107, 82), (134, 165, 135)])

UNITS = "board units: the screen is 1 x 1, x grows to the right, y grows downward; speeds per second"

PHYSICS = {
    "gravity": "the bird's vertical speed grows by 2.45 per second (downward)",
    "max_fall_speed": 1.15,
    "flap": "a flap instantly sets the bird's vertical speed to -0.74 (upward); gravity then slows it",
    "flap_rise": "a flap from rest lifts the bird about 0.11 before it starts falling",
    "ground_y": 0.88,
}


# The same constants as numbers, to simulate the bird (opengames/flap/logic.js).
GRAVITY, LIFT, MAX_FALL, GROUND, HIT_RADIUS, CEILING = 2.45, 0.74, 1.15, 0.88, 0.027, 0.032
SPEED = {"easy": 0.27, "normal": 0.32, "hard": 0.37}  # posts' scroll speed, +0.002 per point up to 40


def _bird(a: np.ndarray) -> tuple[float, float] | None:
    red = (a[..., 0] > 150) & (a[..., 1] < 100) & (a[..., 2] < 70)
    if red.sum() < 50:
        return None
    ys, xs = np.nonzero(red)
    return float(xs.mean()), float(ys.mean())


def detect(frame: Image.Image) -> dict[str, Any]:
    """Pixel positions of the bird and of the nearest post ahead of it (its left/right edges and
    gap), each None when not visible."""
    a = np.asarray(frame.convert("RGB")).astype(int)
    h, w, _ = a.shape
    out: dict[str, Any] = {"size": (w, h), "bird": None, "post": None}
    bird = _bird(a)
    if bird is None:
        return out
    out["bird"] = bird
    red = (a[..., 0] > 150) & (a[..., 1] < 100) & (a[..., 2] < 70)
    bird_left = np.nonzero(red)[1].min()
    post = (np.abs(a[:, :, None, :] - POST_COLORS[None, None]).max(-1) <= 6).any(-1)
    cols = post.sum(0) > h * 0.1
    # Runs of post columns, as (start, end) pairs.
    edges = np.diff(np.concatenate([[0], cols.astype(int), [0]]))
    runs = list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]))
    ahead = [(s, e) for s, e in runs if e > bird_left and e - s > 20]
    if not ahead:
        return out
    s, e = ahead[0]
    rows = post[:, s:e].any(1)
    filled = np.nonzero(rows)[0]
    # The gap is the longest run of empty rows between the post's top and bottom parts.
    empty = np.nonzero(~rows[filled.min():filled.max()])[0] + filled.min()
    if len(empty) == 0:
        return out
    gap = max(np.split(empty, np.nonzero(np.diff(empty) > 1)[0] + 1), key=len)
    out["post"] = {"left": float(s), "right": float(e), "gap_top": float(gap[0]), "gap_bottom": float(gap[-1])}
    return out


class Tracker:
    """Follows the bird across frames `step_ms` apart and reports its height, speed and acceleration."""

    def __init__(self, step_ms: float):
        self.dt = step_ms / 1000
        self.ys: deque[float] = deque(maxlen=3)

    def reset(self) -> None:
        self.ys.clear()

    def update(self, frame: Image.Image) -> dict[str, Any]:
        """Detect everything in `frame`, in board units, and track the bird."""
        d = detect(frame)
        w, h = d["size"]
        bird = None
        if d["bird"]:
            y = d["bird"][1] / h
            self.ys.append(y)
            bird = {"x": round(d["bird"][0] / w, 3), "y": round(y, 3), "vertical_speed": None,
                    "vertical_acceleration": None}
            if len(self.ys) >= 2:
                bird["vertical_speed"] = round((self.ys[-1] - self.ys[-2]) / self.dt, 3)
            if len(self.ys) == 3:
                v1 = (self.ys[1] - self.ys[0]) / self.dt
                v2 = (self.ys[2] - self.ys[1]) / self.dt
                bird["vertical_acceleration"] = round((v2 - v1) / self.dt, 2)
        post = None
        if d["post"]:
            p = d["post"]
            post = {"left_x": round(p["left"] / w, 3), "right_x": round(p["right"] / w, 3),
                    "gap_top_y": round(p["gap_top"] / h, 3), "gap_bottom_y": round(p["gap_bottom"] / h, 3),
                    "gap_middle_y": round((p["gap_top"] + p["gap_bottom"]) / 2 / h, 3)}
        return {"bird": bird, "next_post": post}


def context(measured: dict[str, Any], parts: set[str]) -> dict[str, Any] | None:
    """The `state` for a model: any of "bird" (height, speed, acceleration), "posts" (the next
    post and its gap) and "physics" (the game's constants). None when `parts` is empty."""
    if not parts:
        return None
    out: dict[str, Any] = {"units": UNITS}
    if "bird" in parts:
        out["bird"] = measured["bird"] or "not visible"
    if "posts" in parts:
        out["next_post"] = measured["next_post"] or "no post ahead yet"
    if "physics" in parts:
        out["physics"] = PHYSICS
    return out


def path(y: float, vy: float, seconds: float, dt: float = 1 / 60) -> list[tuple[float, float]]:
    """The bird's (time, y) with no input from height `y` and vertical speed `vy`, as the game steps it."""
    out, t = [(0.0, y)], 0.0
    while t < seconds:
        vy = min(MAX_FALL, vy + GRAVITY * dt)
        y += vy * dt
        if y < CEILING:
            y, vy = CEILING, max(0.0, vy)
        t += dt
        out.append((t, y))
    return out


PATH_COLORS = {"wait": (30, 90, 255), "flap": (255, 150, 0)}  # blue, orange: unlike the bird, posts and sky


def draw_paths(frame: Image.Image, measured: dict[str, Any], step_ms: float, score: int,
               difficulty: str = "easy", seconds: float = 0.3) -> Image.Image:
    """`frame` with the bird's next `seconds` drawn if it doesn't flap (blue) and if it flaps now
    (orange). Each path is as wide as the bird's hitbox and moves right at the posts' scroll speed,
    so it lines up with where the posts will be: a path that touches green or the ground crashes.
    Unchanged when the bird or its speed isn't measured yet."""
    bird = measured["bird"]
    if not bird or bird["vertical_speed"] is None:
        return frame
    # The measured speed is the average over the last step; gravity has added half a step since.
    vy = min(MAX_FALL, bird["vertical_speed"] + GRAVITY * step_ms / 1000 / 2)
    speed = SPEED[difficulty] + min(score, 40) * 0.002
    w, h = frame.size
    out = frame.convert("RGBA")
    layer = Image.new("RGBA", out.size)
    pen = ImageDraw.Draw(layer)
    for name, v in (("wait", vy), ("flap", -LIFT)):
        r = HIT_RADIUS * h
        for t, y in path(bird["y"], v, seconds):
            x, y = (bird["x"] + speed * t) * w, y * h
            pen.ellipse((x - r, y - r, x + r, y + r), fill=PATH_COLORS[name] + (150,))
    return Image.alpha_composite(out, layer).convert("RGB")
