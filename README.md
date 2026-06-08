# Change Detection Pipeline

Compare two equirectangular panoramas, extract aligned perspective patches, and detect visual changes using a vision-language model. The pipeline runs as three Docker services orchestrated by a single Python entry point.

## Quick Start

**Prerequisites:** Docker, NVIDIA GPU (`nvidia-container-toolkit`), Hugging Face model cache.

```bash
# Build images
docker compose build

# Install orchestrator dependency
pip install -r requirements.txt

# Run pipeline
python change_detection_pipeline.py \
  --new-image /path/to/new.jpg \
  --old-image /path/to/old.jpg \
  --metadata metadata.example.json \
  --output-dir ./data/run1 \
  --model-id Qwen/Qwen3.5-9B
```

Results are written to `--output-dir`, including patch images and a `*_batch_results.json` report with detected changes and object names.

## How It Works

| Stage | Image | Description |
|-------|-------|-------------|
| Data prep | `cd-data-prep` | Renders panoramas in a Three.js viewer, captures 6 views, extracts patch pairs |
| Inference | `cd-vllm` | Serves the VLM on port **7100** |
| Detection | `cd-changes` | Compares each patch pair and identifies changes |

The orchestrator (`change_detection_pipeline.py`) runs all three stages sequentially and manages the vLLM container lifecycle.

## Inputs

Two panorama images and a metadata JSON file (`metadata.example.json`):

```json
{
  "new": { "rotation": [rx, ry, rz], "position": [x, y, z] },
  "old": { "rotation": [rx, ry, rz], "position": [x, y, z] }
}
```

`rotation` aligns each panorama in the viewer. `position` is recorded in the output metadata.

## Output

```
output-dir/
├── patch_left_<name>_1..6.png    # NEW patches
├── patch_right_<name>_1..6.png   # OLD patches
└── <name>_06_<model>_batch_results.json
```

Each result entry reports `change_type` (`added`, `removed`, `modified`), `object_name`, and supporting details.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_ID` | `Qwen/Qwen3.5-9B` | Hugging Face model |
| `VLLM_PORT` | `7100` | Inference server port |
| `VLLM_MANAGE_SERVER` | `1` | Orchestrator starts/stops vLLM |
| `HF_HOME` | `~/.cache/huggingface` | Model weight cache |

**Pipeline flags:** `--skip-data-prep`, `--skip-inference`, `--count`, `--step`, `--width`, `--height`.

To run vLLM independently: `MODEL_ID=Qwen/Qwen3.5-9B docker compose up -d vllm`

## License

The default model [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) is licensed under Apache 2.0. Other Hugging Face models carry their own licenses — review the model card before use. Pipeline dependencies (vLLM, Playwright, Three.js, OpenCV) are open source under Apache 2.0 or MIT.
