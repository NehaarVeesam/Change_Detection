# Change Detection Pipeline

Compare two equirectangular panoramas, extract aligned perspective patches, detect visual changes with a vision-language model, and segment changed objects on the panorama using Grounded-DINO + SAM2. The pipeline runs as four Docker services orchestrated by a single Python entry point.

## Quick Start

**Prerequisites:** Docker, NVIDIA GPU (`nvidia-container-toolkit`), Hugging Face model cache.

```bash
# Build images (gdsam image downloads SAM2 + GroundingDINO weights at build time)
docker compose build

# Install orchestrator dependency
pip install -r requirements.txt

# Run full pipeline
python change_detection_pipeline.py \
  --new-image /path/to/new.jpg \
  --old-image /path/to/old.jpg \
  --metadata metadata.json \
  --output-dir ./data/run1 \
  --model-id Qwen/Qwen3.5-9B
```

Or use the provided helper script (update paths as needed):

```bash
bash run.sh
```

## How It Works

| Stage | Image | Description |
|-------|-------|-------------|
| Data prep | `cd-data-prep` | Renders panoramas in a Three.js viewer, captures 6 views, extracts patch pairs |
| Inference | `cd-vllm` | Serves the VLM on port **7100** |
| Detection | `cd-changes` | Compares each patch pair and identifies changes |
| Segmentation | `cd-gdsam` | Uses VLM object names + Grounded-DINO + SAM2 to produce pano masks |

The orchestrator (`change_detection_pipeline.py`) runs all four stages sequentially and manages the vLLM container lifecycle.

## Inputs

Two panorama images and a metadata JSON file (`metadata.json`):

```json
{
  "new": { "rotation": [rx, ry, rz], "position": [x, y, z] },
  "old": { "rotation": [rx, ry, rz], "position": [x, y, z] }
}
```

`rotation` aligns each panorama in the viewer. `position` is recorded in the output metadata.

## Output

After a full run, `--output-dir` contains:

```
output-dir/
├── images/
│   ├── new.jpg                     # copied panoramas
│   └── old.jpg
├── meta.json
├── compare_meta_<name>.json        # per-view corner metadata
├── compare_<name>_1..6.png       # viewer screenshots
├── patch_left_<name>_1..6.png    # NEW patches
├── patch_right_<name>_1..6.png   # OLD patches
├── <name>_06_<model>_batch_results.json
└── gdsam2_<model>_outputs/
    ├── gd_sam2_<model>_annotated.jpg
    ├── gd_sam2_<model>_union_mask.png
    ├── gd_sam2_<model>_masked.jpg
    └── gd_sam2_<model>_annotations.json
```

Each batch result entry reports `change_type` (`added`, `removed`, `modified`), `object_name`, and supporting details. The GD-SAM stage only processes views that have detected changes.

## Running Stages Individually

```bash
# Data prep only
docker compose run --rm data-prep --new-image ... --old-image ... --metadata ... --output-dir /data

# Change detection (expects vLLM running)
docker compose up -d vllm
OUTPUT_DIR=./data/run1 docker compose run --rm changes

# GD-SAM segmentation only (on existing output folder)
OUTPUT_DIR=./data/run1 docker compose run --rm gdsam
```

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_ID` | `Qwen/Qwen3.5-9B` | Hugging Face model |
| `VLLM_PORT` | `7100` | Inference server port |
| `VLLM_MANAGE_SERVER` | `1` | Orchestrator starts/stops vLLM |
| `HF_HOME` | `~/.cache/huggingface` | Model weight cache |
| `OUTPUT_DIR` | `./data/output` | Mount path for compose services |

**Pipeline flags:** `--skip-data-prep`, `--skip-inference`, `--count`, `--step`, `--width`, `--height`.

To run vLLM independently: `MODEL_ID=Qwen/Qwen3.5-9B docker compose up -d vllm`

## GD-SAM Notes

- The `cd-gdsam` image clones [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) and downloads required checkpoints during `docker compose build gdsam`.
- At runtime it reads `--folder /data` (the same output directory used by data-prep and cd-changes).
- No extra env vars are required for this stage.

## License

The default model [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) is licensed under Apache 2.0. Other Hugging Face models carry their own licenses — review the model card before use. Pipeline dependencies (vLLM, Playwright, Three.js, OpenCV, Grounded-SAM-2) are open source under Apache 2.0 or MIT.
