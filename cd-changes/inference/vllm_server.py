"""Start and wait for a local vLLM OpenAI-compatible server."""

from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Optional

from inference.model_registry import vllm_serve_config_from_env


def wait_for_vllm(port: int, timeout_s: float = 1800.0, poll_interval_s: float = 2.0) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(poll_interval_s)
    raise TimeoutError(f"vLLM server not healthy at {url} after {timeout_s}s")


def start_vllm_server(model_id: str, log_path: Optional[str] = None) -> subprocess.Popen:
    cfg = vllm_serve_config_from_env(model_id)
    cmd = cfg.build_command()
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/root/.cache/huggingface")
    stdout = subprocess.DEVNULL
    stderr = subprocess.DEVNULL
    if log_path:
        log_fh = open(log_path, "a", encoding="utf-8")
        stdout = log_fh
        stderr = log_fh
    return subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr)


def ensure_vllm_server(model_id: str) -> None:
    if os.getenv("VLLM_START_SERVER", "1").strip() not in ("1", "true", "yes", "on"):
        port = int(os.getenv("VLLM_PORT", "7100"))
        wait_for_vllm(port)
        return

    port = int(os.getenv("VLLM_PORT", "7100"))
    log_path = os.getenv("VLLM_SERVER_LOG")
    start_vllm_server(model_id, log_path=log_path)
    wait_for_vllm(port)
