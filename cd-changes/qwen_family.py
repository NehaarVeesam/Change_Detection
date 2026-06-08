"""
Compare patch pairs in a single folder via vLLM or transformers.

Authors:
Nehaar Veesam
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from inference.factory import RunnerConfig, create_backend, runner_config_from_env
from inference.vllm_server import ensure_vllm_server
from patch_utils import discover_patch_pairs


def run_folder(cfg: RunnerConfig) -> None:
    folder = cfg.folder.resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    pairs = discover_patch_pairs(folder)
    if not pairs:
        raise FileNotFoundError(
            f"No matched (patch_left, patch_right) pairs in {folder}"
        )

    model_name = cfg.model_id.split("/")[-1]
    last_base, last_idx = pairs[-1][2], pairs[-1][3]
    output_json = folder / f"{last_base}_{last_idx:02d}_{model_name}_batch_results.json"
    if output_json.exists():
        print(f"Results already exist: {output_json}")
        return

    backend = create_backend(cfg)
    backend.warmup()

    results: List[Dict[str, Any]] = []
    for new_a, old_b, base, idx in pairs:
        out = backend.infer_pair(new_a, old_b)
        results.append({"pair": f"{base}_{idx:02d}", **out})

    output_json.write_text(
        json.dumps(
            {
                "model": cfg.model_id,
                "family": backend.family,
                "backend": cfg.backend,
                "folder": str(folder),
                "count": len(results),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(
        description="Run two-image change detection on patch pairs in a folder."
    )
    ap.add_argument(
        "--folder",
        default="/data",
        help="Folder containing patch_left_* and patch_right_* images",
    )
    ap.add_argument("--max_new_tokens", type=int, default=1400)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cache_dir", type=str, default="/root/.cache/huggingface")
    ap.add_argument("--load_4bit", action="store_true")
    args = ap.parse_args()

    model_id = os.getenv("MODEL_ID", "").strip()
    if not model_id:
        raise SystemExit("MODEL_ID environment variable is required")

    folder = Path(args.folder).resolve()
    flash_attn2 = os.getenv("USE_FLASH_ATTN2", "1")
    cfg = runner_config_from_env(folder, model_id, args, flash_attn2)

    if cfg.backend == "vllm" and os.getenv("VLLM_START_SERVER", "0").strip() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        ensure_vllm_server(model_id)

    run_folder(cfg)


if __name__ == "__main__":
    main()
