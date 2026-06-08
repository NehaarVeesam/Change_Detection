"""
Change Detection Pipeline

Orchestrates:
  1. Data preparation  — extract aligned patch pairs from two panoramas
  2. vLLM server       — model inference backend (optional managed start)
  3. Change detection  — run VLM on patch pairs, write results to output dir

Usage:
  python change_detection_pipeline.py \
    --new-image /path/to/new.jpg \
    --old-image /path/to/old.jpg \
    --metadata /path/to/metadata.json \
    --output-dir ./data/run1 \
    --model-id Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_PREP_IMAGE = "cd-data-prep:latest"
CHANGES_IMAGE = "cd-changes:latest"
VLLM_IMAGE = "cd-vllm:latest"
VLLM_CONTAINER = "cd-vllm"


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def run_command(cmd: list[str], stage: str) -> None:
    print(f"[{stage}] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{stage} failed with exit code {result.returncode}")


def wait_for_vllm_health(port: int, timeout_s: int = 1800) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print(f"vLLM ready at {url}")
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(3)
    raise TimeoutError(f"vLLM not healthy at {url} after {timeout_s}s")


def start_vllm(model_id: str, port: int) -> None:
    subprocess.run(["docker", "rm", "-f", VLLM_CONTAINER], check=False)
    hf_cache = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    vllm_extra = os.getenv("VLLM_EXTRA_ARGS", "--max-num-seqs 24")
    cmd = [
        "docker",
        "run",
        "-d",
        "--gpus",
        "all",
        "--network",
        "host",
        "--shm-size=16g",
        "--name",
        VLLM_CONTAINER,
        "-v",
        f"{hf_cache}:/root/.cache/huggingface",
        "-e",
        f"MODEL_ID={model_id}",
        "-e",
        f"VLLM_PORT={port}",
        "-e",
        f"VLLM_MAX_MODEL_LEN={os.getenv('VLLM_MAX_MODEL_LEN', '16384')}",
        "-e",
        f"VLLM_GPU_MEMORY_UTILIZATION={os.getenv('VLLM_GPU_MEMORY_UTILIZATION', '0.93')}",
        "-e",
        f"VLLM_EXTRA_ARGS={vllm_extra}",
        VLLM_IMAGE,
    ]
    run_command(cmd, "Start_vLLM")
    wait_for_vllm_health(port)


def stop_vllm() -> None:
    subprocess.run(["docker", "rm", "-f", VLLM_CONTAINER], check=False)


def run_data_prep(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    new_image = Path(args.new_image).resolve()
    old_image = Path(args.old_image).resolve()
    metadata = Path(args.metadata).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{output_dir}:/data",
        "-v",
        f"{new_image}:{new_image}:ro",
        "-v",
        f"{old_image}:{old_image}:ro",
        "-v",
        f"{metadata}:{metadata}:ro",
        DATA_PREP_IMAGE,
        "--new-image",
        str(new_image),
        "--old-image",
        str(old_image),
        "--metadata",
        str(metadata),
        "--output-dir",
        "/data",
        "--count",
        str(args.count),
        "--step",
        str(args.step),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
    ]
    run_command(cmd, "Data_Prep")


def run_change_detection(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    port = int(os.getenv("VLLM_PORT", "7100"))
    vllm_base_url = os.getenv("VLLM_BASE_URL", f"http://127.0.0.1:{port}/v1")

    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "-v",
        f"{output_dir}:/data",
        "-e",
        f"MODEL_ID={args.model_id}",
        "-e",
        f"INFERENCE_BACKEND={os.getenv('INFERENCE_BACKEND', 'vllm')}",
        "-e",
        "VLLM_START_SERVER=0",
        "-e",
        f"VLLM_BASE_URL={vllm_base_url}",
        "-e",
        f"ENABLE_THINKING={os.getenv('ENABLE_THINKING', '0')}",
        CHANGES_IMAGE,
        "--folder",
        "/data",
    ]
    run_command(cmd, "Change_Detection")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the change detection pipeline")
    parser.add_argument("--new-image", required=True, help="Path to the NEW panorama image")
    parser.add_argument("--old-image", required=True, help="Path to the OLD panorama image")
    parser.add_argument(
        "--metadata",
        required=True,
        help="JSON file with rotation and position for new and old images",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "data" / "output"),
        help="Directory for patches and inference results",
    )
    parser.add_argument("--model-id", required=True, help="Vision-language model ID")
    parser.add_argument("--count", type=int, default=6, help="Number of patch views")
    parser.add_argument("--step", type=float, default=-60.0, help="Yaw step in degrees")
    parser.add_argument("--width", type=int, default=3840, help="Screenshot width")
    parser.add_argument("--height", type=int, default=1080, help="Screenshot height")
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip patch extraction if patches already exist in output-dir",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Only run data preparation",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    for path_str, label in (
        (args.new_image, "--new-image"),
        (args.old_image, "--old-image"),
        (args.metadata, "--metadata"),
    ):
        if not Path(path_str).exists():
            raise SystemExit(f"{label} not found: {path_str}")


def run_pipeline(args: argparse.Namespace) -> None:
    load_dotenv()
    validate_inputs(args)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if not args.skip_data_prep:
        run_data_prep(args)
    else:
        print("Skipping data preparation")

    if args.skip_inference:
        print("Skipping change detection inference")
        return

    port = int(os.getenv("VLLM_PORT", "7100"))
    manage_vllm = _env_truthy("VLLM_MANAGE_SERVER", "1")

    try:
        if manage_vllm:
            start_vllm(args.model_id, port)
        else:
            print(f"Expecting existing vLLM at port {port}")
            wait_for_vllm_health(port)

        run_change_detection(args)
        print(f"Pipeline complete. Results in {output_dir}")
    finally:
        if manage_vllm:
            stop_vllm()


if __name__ == "__main__":
    try:
        run_pipeline(parse_arguments())
    except KeyboardInterrupt:
        sys.exit(130)
