"""Model family detection and vLLM serve configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ModelFamily:
    key: str
    label: str


FAMILIES = {
    "qwen35": ModelFamily("qwen35", "Qwen3.5 (unified VL)"),
    "qwen3_vl": ModelFamily("qwen3_vl", "Qwen3-VL"),
    "qwen25_vl": ModelFamily("qwen25_vl", "Qwen2.5-VL"),
}


def detect_family(model_id: str) -> str:
    if "Qwen3.5" in model_id or "Qwen3_5" in model_id:
        return "qwen35"
    if "Qwen3-VL" in model_id:
        return "qwen3_vl"
    if "Qwen2.5-VL" in model_id or "Qwen2_5-VL" in model_id:
        return "qwen25_vl"
    raise ValueError(
        f"Unsupported model_id: {model_id}. "
        "Supported: Qwen3.5-*, Qwen3-VL-*, Qwen2.5-VL-*"
    )


@dataclass
class VllmServeConfig:
    model_id: str
    family: str
    port: int = 7100
    tensor_parallel_size: int = 1
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    extra_args: List[str] = field(default_factory=list)

    def build_command(self) -> List[str]:
        cmd = [
            "vllm",
            "serve",
            self.model_id,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
        ]
        if self.family == "qwen35":
            cmd.extend(["--reasoning-parser", "qwen3", "--trust-remote-code"])
        elif self.family in ("qwen3_vl", "qwen25_vl"):
            cmd.extend(
                [
                    "--limit-mm-per-prompt",
                    '{"image": 10}',
                    "--trust-remote-code",
                ]
            )
        cmd.extend(self.extra_args)
        return cmd


def vllm_serve_config_from_env(model_id: str) -> VllmServeConfig:
    family = detect_family(model_id)
    port = int(os.getenv("VLLM_PORT", "7100"))
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "32768"))
    gpu_mem = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
    tp = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
    extra: List[str] = []
    extra_env = os.getenv("VLLM_EXTRA_ARGS", "").strip()
    if extra_env:
        extra.extend(extra_env.split())
    if family == "qwen35" and os.getenv("VLLM_MAX_MODEL_LEN") is None:
        max_model_len = min(max_model_len, 32768)
    return VllmServeConfig(
        model_id=model_id,
        family=family,
        port=port,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        extra_args=extra,
    )
