"""vLLM OpenAI-compatible API client."""

from __future__ import annotations

import base64
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from inference.backends.base import InferenceBackend
from inference.model_registry import detect_family
from patch_utils import json_from_text, normalize_and_filter
from prompts import PROMPT_PAIR_DIRECT, SYSTEM_INSTRUCTION


def _image_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if not mime:
        mime = "image/png"
    raw = path.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_messages(img_a: Path, img_b: Path) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _image_to_data_url(img_a)}},
                {"type": "image_url", "image_url": {"url": _image_to_data_url(img_b)}},
                {"type": "text", "text": PROMPT_PAIR_DIRECT},
            ],
        },
    ]


@dataclass
class VllmBackendConfig:
    model_id: str
    base_url: str
    api_key: str
    max_new_tokens: int
    top_p: float
    enable_thinking: bool
    temperature: float = 0.7
    request_timeout_s: float = 600.0
    max_retries: int = 3


class VllmBackend(InferenceBackend):
    def __init__(self, cfg: VllmBackendConfig):
        self.cfg = cfg
        self._family = detect_family(cfg.model_id)
        self.client = OpenAI(
            base_url=cfg.base_url.rstrip("/"),
            api_key=cfg.api_key or "EMPTY",
            timeout=cfg.request_timeout_s,
        )

    @property
    def family(self) -> str:
        return self._family

    def _extra_body(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if self._family == "qwen35" and not self.cfg.enable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    def _completion(self, img_a: Path, img_b: Path) -> str:
        extra = self._extra_body()
        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.cfg.model_id,
                    messages=_build_messages(img_a, img_b),
                    max_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    extra_body=extra if extra else None,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_err = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(
            f"vLLM request failed after {self.cfg.max_retries} attempts: {last_err}"
        ) from last_err

    def infer_pair(self, img_a: Path, img_b: Path) -> Dict[str, Any]:
        text_out = self._completion(img_a, img_b)
        return normalize_and_filter(json_from_text(text_out))

    def warmup(self) -> None:
        models = self.client.models.list()
        if not models.data:
            raise RuntimeError(f"No models listed at {self.cfg.base_url}")
