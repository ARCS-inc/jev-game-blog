"""One interface for System One decision models, local (d1 on Hugging Face) or hosted (Jev API).

Questions are plain dicts in the System One format shared by both, `{"type": "noul" | "choice" |
"score", "instructions": ..., "criteria": ...}`, keyed by name, so prompts can live outside the
code (see `load_questions`). Both backends answer the same way:

    model = load_model("local:LiquidAI/d1-3B")      # or "api:jev-latest"
    questions = load_questions("prompts/above_below.json")
    d = model.predict(frame, questions)               # one image
    d = model.predict_frames([f0, f1, f2], questions) # several images, oldest first
    d.answers["act"]["probabilities"]["below"], d.latency_ms
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

Questions = Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class Decision:
    answers: dict[str, dict[str, Any]]
    """Answers by question name, e.g. `{"type": "choice", "choice": ..., "probabilities": {...}}`."""
    latency_ms: float
    """Wall time of the call: one forward pass locally, the HTTP round trip for the API."""
    input_tokens: int | None
    model: str
    """The model that answered; for the API, the versioned ID it reports (e.g. jev-1.13.0)."""


def load_questions(path: str | Path) -> dict[str, dict[str, Any]]:
    """Questions from a JSON file: either one question (named after the file) or a mapping of them."""
    path = Path(path)
    data = json.loads(path.read_text())
    return {path.stem: data} if "type" in data else data


class DecisionModel(ABC):
    name: str
    threshold: float = 0.5
    """Default P(flap) at or above which game.play flaps."""
    history: int = 1
    """Default number of recent frames game.play shows the model (1 = the current frame only)."""

    def set_defaults(self, threshold: float | None = None, history: int | None = None) -> None:
        """Override the class defaults for this instance."""
        if threshold is not None:
            self.threshold = threshold
        if history is not None:
            self.history = history

    def predict(self, image: Image.Image, questions: Questions, state: Any = None) -> Decision:
        """Answer `questions` about one image (and optional extra `state`)."""
        return self._timed([image], questions, state)

    def predict_frames(self, images: Sequence[Image.Image], questions: Questions, state: Any = None) -> Decision:
        """Answer `questions` about several images in order, e.g. the last few frames, oldest first."""
        if not images:
            raise ValueError("predict_frames needs at least one image")
        return self._timed(list(images), questions, state)

    def _timed(self, images: list[Image.Image], questions: Questions, state: Any) -> Decision:
        if not questions:
            raise ValueError("no questions")
        t = time.perf_counter()
        answers, tokens, model = self._run(images, questions, state)
        return Decision(answers, (time.perf_counter() - t) * 1000, tokens, model or self.name)

    @abstractmethod
    def _run(self, images: list[Image.Image], questions: Questions,
             state: Any) -> tuple[dict[str, dict[str, Any]], int | None, str | None]: ...


class LocalModel(DecisionModel):
    """A d1 checkpoint run with transformers (`trust_remote_code`), on CUDA, MPS or CPU."""

    def __init__(self, repo: str = "LiquidAI/d1-3B", device: str | None = None, revision: str | None = None):
        import torch
        from transformers import AutoModel

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(
            repo, trust_remote_code=True, revision=revision,
            dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
        ).to(self.device)
        self.name = repo

    def _run(self, images, questions, state):
        out = self.model.system_one(state, dict(questions), images=images)
        if self.device == "cuda":
            self._torch.cuda.synchronize()
        return out["answers"], out.get("usage", {}).get("input_tokens"), None


class ApiModel(DecisionModel):
    """A hosted System One endpoint through the TypeSafe SDK: Jev by default, or any compatible API
    (pass its `base_url`). The key comes from `api_key` or `TYPESAFE_API_KEY`."""

    def __init__(self, model: str = "jev-latest", base_url: str | None = None, api_key: str | None = None,
                 image_format: str = "png", headers: Mapping[str, str] | None = None, timeout: float = 30.0):
        from typesafe_sdk import TypeSafeClient

        # The API rejects images unless this opt-in header is sent (they are experimental).
        headers = {"TypeSafe-Danger-Zone-Images": "true", **(headers or {})}
        self.client = TypeSafeClient(api_key=api_key, model=model, base_url=base_url,
                                     headers=headers, timeout=timeout)
        self.name = model
        self.image_format = image_format

    def _run(self, images, questions, state):
        from typesafe_sdk import Image as AssetImage

        assets = [AssetImage.from_pil(im.convert("RGB"), format=self.image_format) for im in images]
        pics: Any = assets[0] if len(assets) == 1 else assets
        payload = pics if state is None else {"state": state, "images": pics}
        res = self.client.system_one(state=payload, questions=dict(questions))  # type: ignore[arg-type]  # raw dicts are accepted
        answers = {k: a.model_dump(mode="json") for k, a in res.answers.items()}
        return answers, res.usage.input_tokens if res.usage else None, res.model

    def close(self) -> None:
        self.client.close()


class D1Model(LocalModel):
    """d1 run locally. It reads one frame well but can't compare frames (05: motion AUC ~0.55 on 1-4
    frames), so it sees the current frame only."""

    threshold = 0.5
    history = 1

    def __init__(self, repo: str = "LiquidAI/d1-3B", *, threshold: float | None = None,
                 history: int | None = None, **kwargs):
        super().__init__(repo, **kwargs)
        self.set_defaults(threshold, history)


class JevModel(ApiModel):
    """Jev through the TypeSafe API. It tells rising from falling across frames (06: motion AUC 0.89
    on 4 frames), so it sees the last 4. Its P(flap) for should_flap sat mostly at 0.6-0.78 when the
    bird had to climb (07), so it flaps from 0.6."""

    threshold = 0.6
    history = 4

    def __init__(self, model: str = "jev-latest", *, threshold: float | None = None,
                 history: int | None = None, **kwargs):
        super().__init__(model, **kwargs)
        self.set_defaults(threshold, history)


def load_model(spec: str) -> DecisionModel:
    """`d1:<hf repo>` / `local:<hf repo>` (default LiquidAI/d1-3B) or `jev:<model>` / `api:<model>`
    (default jev-latest); a bare spec is a local repo."""
    kind, _, arg = spec.partition(":") if spec.split(":", 1)[0] in ("local", "d1", "api", "jev") else ("d1", "", spec)
    if kind in ("api", "jev"):
        return JevModel(arg or "jev-latest")
    return D1Model(arg or "LiquidAI/d1-3B")
