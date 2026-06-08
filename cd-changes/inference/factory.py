"""Create inference backend from configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from inference.backends.base import InferenceBackend
from inference.backends.vllm_backend import VllmBackend, VllmBackendConfig


@dataclass
class RunnerConfig:
    folder: Path
    model_id: str
    max_new_tokens: int
    top_p: float
    seed: int
    load_4bit: bool
    cache_dir: Path
    use_flash_attn2: bool
    backend: str
    vllm_base_url: str
    vllm_api_key: str
    enable_thinking: bool


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def create_backend(cfg: RunnerConfig) -> InferenceBackend:
    backend = (cfg.backend or "vllm").strip().lower()
    if backend == "transformers":
        from inference.backends.transformers_backend import (
            TransformersBackend,
            TransformersBackendConfig,
        )

        return TransformersBackend(
            TransformersBackendConfig(
                model_id=cfg.model_id,
                max_new_tokens=cfg.max_new_tokens,
                top_p=cfg.top_p,
                seed=cfg.seed,
                load_4bit=cfg.load_4bit,
                cache_dir=cfg.cache_dir,
                use_flash_attn2=cfg.use_flash_attn2,
            )
        )
    if backend != "vllm":
        raise ValueError(f"Unknown INFERENCE_BACKEND={cfg.backend!r}. Use 'vllm' or 'transformers'.")
    return VllmBackend(
        VllmBackendConfig(
            model_id=cfg.model_id,
            base_url=cfg.vllm_base_url,
            api_key=cfg.vllm_api_key,
            max_new_tokens=cfg.max_new_tokens,
            top_p=cfg.top_p,
            enable_thinking=cfg.enable_thinking,
        )
    )


def runner_config_from_env(folder: Path, model_id: str, args, flash_attn2) -> RunnerConfig:
    backend = os.getenv("INFERENCE_BACKEND", "vllm")
    port = os.getenv("VLLM_PORT", "7100")
    default_base = f"http://127.0.0.1:{port}/v1"
    return RunnerConfig(
        folder=folder,
        model_id=model_id,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        seed=args.seed,
        load_4bit=args.load_4bit,
        cache_dir=Path(args.cache_dir),
        use_flash_attn2=str(flash_attn2).strip() in ("1", "true", "True", "yes", "on"),
        backend=backend,
        vllm_base_url=os.getenv("VLLM_BASE_URL", default_base),
        vllm_api_key=os.getenv("VLLM_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")),
        enable_thinking=_env_bool("ENABLE_THINKING", False),
    )
