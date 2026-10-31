"""A decision model plays OpenGames Flap from screenshots.

The page's clock is frozen with Playwright (`install` alone keeps ticking in real time, so it is
also paused), and the game only advances on `clock.run_for`: each turn is screenshot -> one model
call -> optional flap -> advance `step_ms`. Model latency therefore never costs the bird a frame.
"""

from __future__ import annotations

import functools
import http.server
import io
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from playwright.sync_api import sync_playwright

import json

from jev_flap.harness import Tracker, context, draw_paths
from jev_flap.model import DecisionModel

GAMES_DIR = Path(__file__).resolve().parents[2] / "opengames"


def p_flap(answer: Mapping[str, Any], flap_option: str | None) -> float:
    """P(flap) from an answer: a noul's P(yes) when `flap_option` is None, else that option's probability."""
    return answer["noul"] if flap_option is None else answer["probabilities"][flap_option]


@dataclass
class Episode:
    score: int
    turns: int
    flaps: int


def serve(directory: Path = GAMES_DIR) -> http.server.ThreadingHTTPServer:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def latency_summary(name: str, ms: list[float]) -> str:
    if not ms:
        return f"{name}: no samples"
    q = sorted(ms)
    pct = lambda f: q[min(len(q) - 1, int(f * len(q)))]
    return (f"{name}: n={len(q)} mean={statistics.mean(q):.1f}ms p50={pct(0.5):.1f}ms "
            f"p90={pct(0.9):.1f}ms p99={pct(0.99):.1f}ms max={q[-1]:.1f}ms")


def play(model: DecisionModel, question: Mapping[str, Any], flap_option: str | None,
         threshold: float | None = None, *, out_dir: Path, episodes: int = 3, step_ms: int = 80,
         difficulty: str = "easy", history: int | None = None, max_turns: int = 5000,
         context_parts: frozenset[str] = frozenset(), paths: bool = False,
         policy: Callable[[Mapping[str, Any]], float] | None = None, headed: bool = False) -> list[Episode]:
    """Play `episodes` rounds, flapping when P(flap) >= `threshold`. The model sees the last `history`
    frames (oldest first; the first turns repeat the opening frame). Both default to the model's
    own (`model.threshold`, `model.history`). `context_parts` (any of "bird", "posts", "physics")
    are measured from each frame and passed as the model's `state` (see harness.context). With
    `paths`, the bird's paths if it waits or flaps are drawn on the frame (see harness.draw_paths).
    With `policy`, `question` is a mapping of named questions and `policy` turns their answers into
    the score compared with `threshold` (`flap_option` is unused). Writes `run.log` and one
    `ep<N>.gif` of the frames the model saw per episode to `out_dir`."""
    threshold = model.threshold if threshold is None else threshold
    history = model.history if history is None else history
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = (out_dir / "run.log").open("w")

    def log(line: str) -> None:
        print(line, flush=True)
        log_file.write(line + "\n")

    log(f"model {model.name} threshold {threshold} history {history} step {step_ms}ms difficulty {difficulty} "
        f"context {sorted(context_parts) or 'none'}" + (" paths" if paths else ""))
    questions = question if policy else {"act": question}
    decide = policy or (lambda answers: p_flap(answers["act"], flap_option))
    tracker = Tracker(step_ms)

    server = serve()
    url = f"http://127.0.0.1:{server.server_address[1]}/flap/index.html"
    results, shot_ms, model_ms, tokens = [], [], [], 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed)
        page = browser.new_page(viewport={"width": 800, "height": 900})
        page.clock.install()  # fake requestAnimationFrame / performance.now; still ticks in real time
        page.add_init_script(f"localStorage.setItem('opengames:flap:difficulty', {difficulty!r})")
        page.goto(url)
        page.clock.pause_at(page.evaluate("Date.now()") + 1000)  # freeze: only run_for advances the game
        canvas = page.locator("#board")
        overlay = page.locator("#overlay")

        for ep in range(episodes):
            page.keyboard.press("r")  # fresh round, already started: the bird hovers until the first flap
            page.keyboard.press("ArrowUp")  # kick off the round so a model that never flaps still falls
            frames, recent, turn, flaps = [], deque(maxlen=history), 0, 0
            tracker.reset()
            while overlay.is_hidden() and turn < max_turns:
                t0 = time.perf_counter()
                shot = Image.open(io.BytesIO(canvas.screenshot())).convert("RGB")
                shot_ms.append((time.perf_counter() - t0) * 1000)
                measured = tracker.update(shot)
                if paths:
                    shot = draw_paths(shot, measured, step_ms, int(page.locator("#score").inner_text()), difficulty)
                if not recent:
                    recent.extend([shot] * history)
                recent.append(shot)
                state = context(measured, context_parts)
                d = (model.predict(shot, questions, state) if history == 1
                     else model.predict_frames(list(recent), questions, state))
                model_ms.append(d.latency_ms)
                tokens += d.input_tokens or 0
                pf = decide(d.answers)
                if pf >= threshold:
                    page.keyboard.press("ArrowUp")  # unlike Space, not swallowed by a focused button
                    flaps += 1
                frames.append(shot)
                log(f"ep{ep} t{turn:4d} p(flap)={pf:.3f} {'FLAP' if pf >= threshold else '    '} "
                    f"model={d.latency_ms:6.1f}ms shot={shot_ms[-1]:5.1f}ms score={page.locator('#score').inner_text()}"
                    + "".join(f" {k}={a['noul']:.3f}" for k, a in d.answers.items() if policy and "noul" in a)
                    + (f" state={json.dumps({k: v for k, v in state.items() if k not in ('units', 'physics')})}" if state else ""))
                page.clock.run_for(step_ms)
                turn += 1
            score = int(page.locator("#score").inner_text())
            results.append(Episode(score, turn, flaps))
            log(f"episode {ep}: score {score} ({turn} turns, {flaps} flaps)")
            frames[0].save(out_dir / f"ep{ep}.gif", save_all=True, append_images=frames[1:],
                           duration=step_ms, loop=0)
        browser.close()
    server.shutdown()
    # The first call includes CUDA warm-up or connection setup, so report it separately.
    if model_ms:
        log(f"first model call: {model_ms[0]:.1f}ms")
    log(latency_summary(f"model {model.name} (excl. first)", model_ms[1:]))
    log(latency_summary("screenshot", shot_ms))
    log(f"input tokens: {tokens}")
    log(f"scores: {[r.score for r in results]}")
    log_file.close()
    return results
