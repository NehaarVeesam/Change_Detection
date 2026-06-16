'''
Grounded-DINO + SAM2 segmentation on change-detection outputs.
'''

import re
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import cv2
import torch
import numpy as np
import pycocotools.mask as mask_util
import supervision as sv

DATA_DIR = "/data"

# ------------------ CONFIG DEFAULTS ------------------ #

DEFAULT_GSAM2_ROOT = "/home/ubuntu/workdir/Grounded-SAM-2"

SAM2_CFG_REL = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT_REL = "checkpoints/sam2.1_hiera_large.pt"

GDINO_CFG_REL = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CKPT_REL = "gdino_checkpoints/groundingdino_swint_ogc.pth"

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
MULTIMASK_OUTPUT = True
BEST_SINGLE_BOX_PER_CHANGE = True
# If True, use full-height vertical strips per view instead of tight quad band
FULL_VERTICAL_STRIP = False

# ------------------ UTILS ------------------ #
gdino_load_model = None
gdino_load_image = None
gdino_predict = None


def _ensure_gsam2_path(gs_root: Path) -> None:
    """Ensure Grounded-SAM-2 repo root is available on sys.path."""
    gs_root_str = str(gs_root.resolve())
    if gs_root_str not in sys.path:
        sys.path.append(gs_root_str)


def single_mask_to_rle(mask: np.ndarray) -> dict:
    """Convert a single boolean mask (H, W) to COCO RLE dict."""
    rle = mask_util.encode(
        np.array(mask[:, :, None], order="F", dtype="uint8")
    )[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def build_sam2_predictor(gs_root: Path, device: str = "cuda") -> Any:
    _ensure_gsam2_path(gs_root)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    cfg_str = SAM2_CFG_REL
    ckpt_path = gs_root / SAM2_CKPT_REL
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt_path}")

    print(f"[SAM2] cfg (string): {cfg_str}")
    print(f"[SAM2] ckpt:        {ckpt_path}")

    model = build_sam2(cfg_str, str(ckpt_path), device=device)
    return SAM2ImagePredictor(model)


def build_gdino_model(gs_root: Path, device: str = "cuda"):
    global gdino_load_model, gdino_load_image, gdino_predict
    _ensure_gsam2_path(gs_root)
    from grounding_dino.groundingdino.util.inference import (
        load_model as _gdino_load_model,
        load_image as _gdino_load_image,
        predict as _gdino_predict,
    )
    gdino_load_model = _gdino_load_model
    gdino_load_image = _gdino_load_image
    gdino_predict = _gdino_predict

    cfg_path = gs_root / GDINO_CFG_REL
    ckpt_path = gs_root / GDINO_CKPT_REL

    if not cfg_path.is_file():
        raise FileNotFoundError(f"GroundingDINO config not found: {cfg_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"GroundingDINO checkpoint not found: {ckpt_path}")

    print(f"[G-DINO] cfg:  {cfg_path}")
    print(f"[G-DINO] ckpt: {ckpt_path}")

    model = gdino_load_model(
        model_config_path=str(cfg_path),
        model_checkpoint_path=str(ckpt_path),
        device=device,
    )
    return model

_BAD_TOKENS = {
    "left","right","top","bottom","center","centre","middle",
    "near","next","along","adjacent","beside","between","across",
    "foreground","background","image","photo","view",
    "installed","install","constructed","complete","completed","partial","started",
    "new","old","latest","earlier",
}

def normalize_prompt(text: str) -> str:

    text = (text or "").strip().lower()
    text = re.sub(r"\([^)]*\)", " ", text)             
    text = re.sub(r"[^a-z0-9\s\-\/\.]", " ", text)     
    text = re.sub(r"\s+", " ", text).strip()

    text = re.sub(r"\s*\.\s*", " . ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text.endswith("."):
        text += "."
    return text

def _clean_term(t: str) -> str:
    t = (t or "").strip().lower()
    t = re.sub(r"\([^)]*\)", " ", t)
    t = re.sub(r"[^a-z0-9\s\-\/]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _is_bad_term(t: str) -> bool:
    toks = t.split()
    if not toks:
        return True
    return all(tok in _BAD_TOKENS for tok in toks)

def build_text_prompts(change: Dict) -> List[str]:
    """
    Build GroundingDINO captions from change-detection output:
      - object_name (primary)
      - object_names_alternatives (synonyms)
    """
    target = (
        change.get("object_name")
        or change.get("target_object")
        or change.get("object_label")
        or ""
    ).strip()
    if not target:
        return ["construction object."]

    alts = (
        change.get("object_names_alternatives")
        or change.get("caption_alternatives")
        or []
    )
    if not isinstance(alts, list):
        alts = []

    raw_terms = [target] + alts

    seen = set()
    terms: List[str] = []
    for t in raw_terms:
        t2 = _clean_term(str(t))
        if not t2 or _is_bad_term(t2):
            continue
        if t2 in seen:
            continue
        seen.add(t2)
        terms.append(t2)
        if len(terms) >= 6: 
            break

    tgt = _clean_term(target)
    if tgt and (not terms or terms[0] != tgt):
        terms = [tgt] + [x for x in terms if x != tgt]
    prompts: List[str] = []

    if terms:
        prompts.append(normalize_prompt(" . ".join(terms)))
    prompts.append(normalize_prompt(tgt))

    if len(terms) >= 2:
        prompts.append(normalize_prompt(" . ".join(terms[:2])))

    uniq: List[str] = []
    seen_p = set()
    for p in prompts:
        if p and p not in seen_p:
            seen_p.add(p)
            uniq.append(p)

    return uniq


def compute_patch_bounds(corners_left: dict, pano_w: int, pano_h: int):
    """
    If FULL_VERTICAL_STRIP:
        - use only TL/TR x coordinates
        - y from 0 .. pano_h-1  (full vertical strip)
    Else:
        - use TL/TR/BL/BR x,y and make a tight rectangle around the quad.
    """
    tl = corners_left["tl"]
    tr = corners_left["tr"]
    bl = corners_left["bl"]
    br = corners_left["br"]

    texW = int(tl["texWidth"])
    texH = int(tl["texHeight"])

    if pano_w != texW or pano_h != texH:
        print(f"[WARN] pano size ({pano_w},{pano_h}) != meta tex ({texW},{texH}); "
              f"assuming they are aligned.")

    if FULL_VERTICAL_STRIP:
        xs = np.array([tl["x"], tr["x"]], dtype=float)

        span = xs.max() - xs.min()
        if span <= texW / 2:
            xs_un = xs
        else:
            xs_un = np.where(xs > texW / 2, xs - texW, xs)

        xmin_un = int(np.floor(xs_un.min()))
        xmax_un = int(np.ceil(xs_un.max()))

        ymin = 0
        ymax = pano_h - 1

        patch_w = xmax_un - xmin_un + 1
        patch_h = ymax - ymin + 1
    else:
        xs = np.array([tl["x"], tr["x"], bl["x"], br["x"]], dtype=float)
        ys = np.array([tl["y"], tr["y"], bl["y"], br["y"]], dtype=float)

        span = xs.max() - xs.min()
        if span <= texW / 2:
            xs_un = xs
        else:
            xs_un = np.where(xs > texW / 2, xs - texW, xs)

        xmin_un = int(np.floor(xs_un.min()))
        xmax_un = int(np.ceil(xs_un.max()))
        ymin = int(np.floor(ys.min()))
        ymax = int(np.ceil(ys.max()))

        patch_w = xmax_un - xmin_un + 1
        patch_h = ymax - ymin + 1

    return xmin_un, ymin, patch_w, patch_h, texW, texH


def extract_patch_from_pano(pano_bgr: np.ndarray, corners_left: dict):
    """
    Build a rectangular patch from pano using seam-safe bounds.

    Returns:
      patch_bgr, (xmin_unwrapped, ymin)
    """
    pano_h, pano_w = pano_bgr.shape[:2]
    xmin_un, ymin, patch_w, patch_h, texW, texH = compute_patch_bounds(
        corners_left, pano_w, pano_h
    )

    patch = np.zeros((patch_h, patch_w, 3), dtype=pano_bgr.dtype)

    for dy in range(patch_h):
        yp = ymin + dy
        if yp < 0 or yp >= pano_h:
            continue
        xs_un = xmin_un + np.arange(patch_w)
        xs = xs_un % pano_w
        patch[dy, :] = pano_bgr[yp, xs]

    return patch, (xmin_un, ymin)

def _pick_latest_file(paths):
    return max(paths, key=lambda p: p.stat().st_mtime)

def load_latest_batch_results(folder: Path) -> dict:

    cands = list(folder.glob("*_batch_results.json"))
    if not cands:
        raise FileNotFoundError(f"No *_batch_results.json found in {folder}")
    latest = _pick_latest_file(cands)
    return json.loads(latest.read_text(encoding="utf-8"))

def build_pair_to_result(batch: dict) -> dict:
    m = {}
    for r in batch.get("results", []):
        pair = r.get("pair")
        if pair:
            m[pair] = r 
    return m

# ------------------ CORE ------------------ #

def process_view_on_pano(
    idx: int,
    view_key: str,
    meta_entry: dict,
    pano_bgr: np.ndarray,
    qwen_result: dict,
    sam2_predictor: Any,
    gdino_model,
    device: str = "cuda",
    folder: Path | None = None,
    base_id: str | None = None,
):

    """
    Run GroundingDINO+SAM2 for one compare_i view directly on pano:

    - crop patch from pano using corners.left
    - run G-DINO+SAM2 on patch using text from Qwen JSON
    - map masks back to global pano coords
    """
    print(f"\n=== View #{idx} → {view_key} ===")
    if isinstance(qwen_result, dict) and "result" in qwen_result and isinstance(qwen_result["result"], dict):
        result = qwen_result["result"]
    else:
        result = qwen_result if isinstance(qwen_result, dict) else {}


    changes = result.get("changes", []) or []
    if not changes:
        print("No changes found in JSON; skipping.")
        return [], [], [], [], []


    corners_left = meta_entry["corners"]["left"]
    patch_bgr, (xmin_un, ymin) = extract_patch_from_pano(pano_bgr, corners_left)
    patch_h, patch_w = patch_bgr.shape[:2]
    print(f"Patch size: {patch_w}x{patch_h}")

    idx_int = int(idx)
    patch_path = folder / f"patch_left_{base_id}_{idx_int}.png"
    if patch_path.exists():
        print(f"Using existing patch: {patch_path}")
    else:
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(patch_path), patch_bgr):
            print(f"[WARN] Failed to write patch image: {patch_path}")
            return [], [], [], [], []

    image_source, image = gdino_load_image(str(patch_path))  # RGB HxWx3
    h, w, _ = image_source.shape

    if (w, h) != (patch_w, patch_h):
        print("[WARN] GroundingDINO resized patch; using its size for boxes/masks")
        patch_w, patch_h = w, h

    sam2_predictor.set_image(image_source)

    use_cuda = (device == "cuda" and torch.cuda.is_available())
    if use_cuda and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from torchvision.ops import box_convert

    pano_h, pano_w = pano_bgr.shape[:2]

    all_boxes_global = []
    all_masks_global = []
    all_labels = []
    all_class_ids = []
    all_annotations = []
    det_counter = 0

    for change_idx, change in enumerate(changes):
        prompt_candidates = build_text_prompts(change)
        used_prompt = None
        boxes = None
        box_scores = None
        phrases = None

        # text → boxes on PATCH
        for p_idx, prompt in enumerate(prompt_candidates):
            print(f"Change {change_idx}: prompt[{p_idx}] = '{prompt}'")
            boxes_try, scores_try, phrases_try = gdino_predict(
                model=gdino_model,
                image=image,
                caption=prompt,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=device,
            )
            if boxes_try.size(0) == 0:
                print("    G-DINO: no boxes for this prompt.")
                continue

            used_prompt = prompt
            boxes = boxes_try
            box_scores = scores_try
            phrases = phrases_try
            break

        if boxes is None or boxes.size(0) == 0:
            print(f"Change {change_idx}: no boxes for ANY prompt; skipping.")
            continue

        boxes_px = boxes * torch.tensor([patch_w, patch_h, patch_w, patch_h], device=boxes.device)

        if BEST_SINGLE_BOX_PER_CHANGE:
            best_idx = torch.argmax(box_scores).item()
            boxes_px = boxes_px[best_idx:best_idx+1]
            box_scores = box_scores[best_idx:best_idx+1]
            phrases = [phrases[best_idx]]

        input_boxes_xyxy = box_convert(
            boxes=boxes_px,
            in_fmt="cxcywh",
            out_fmt="xyxy",
        ).cpu().numpy()
        if use_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks, scores, logits = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes_xyxy,
                    multimask_output=MULTIMASK_OUTPUT,
                )
        else:
            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes_xyxy,
                multimask_output=MULTIMASK_OUTPUT,
            )

        scores = np.array(scores)

        if MULTIMASK_OUTPUT and scores.ndim == 2:
            best = np.argmax(scores, axis=1)
            masks = masks[np.arange(masks.shape[0]), best]
            scores = scores[np.arange(scores.shape[0]), best]
        elif scores.ndim == 1:
            pass
        else:
            print(f"SAM2: unexpected scores shape {scores.shape}; skipping.")
            continue

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        masks_bool = masks.astype(bool)

        best_idx = None
        best_score = None
        for i, (mask_i, s_i) in enumerate(zip(masks_bool, scores)):
            ys, xs = np.where(mask_i)
            if xs.size == 0 or ys.size == 0:
                continue
            if best_idx is None or s_i > best_score:
                best_idx = i
                best_score = s_i

        if best_idx is None:
            continue

        mask_patch = masks_bool[best_idx]
        score = float(best_score)

        ys, xs = np.where(mask_patch)
        if xs.size == 0 or ys.size == 0:
            continue

        xs_un = xs + xmin_un
        xs_glob = xs_un % pano_w
        ys_glob = ys + ymin

        mask_global = np.zeros((pano_h, pano_w), dtype=bool)
        mask_global[ys_glob, xs_glob] = True

        x1g = float(xs_glob.min())
        x2g = float(xs_glob.max())
        y1g = float(ys_glob.min())
        y2g = float(ys_glob.max())
        bbox_global = [x1g, y1g, x2g, y2g]

        class_name = (
            change.get("object_name")
            or change.get("object_label")
            or change.get("category")
            or "change"
        )
        label = f"{class_name} {score:.2f}"

        rle = single_mask_to_rle(mask_global)

        annotation = {
            "view_index": idx,
            "view_key": view_key,
            "change_index": change_idx,
            "object_label": change.get("object_label"),
            "category": change.get("category"),
            "description": change.get("description"),
            "location": change.get("location"),
            "caption": change.get("caption"),
            "prompt_used": used_prompt,
            "phrase": phrases[0] if phrases else used_prompt,
            "box_score": float(box_scores[0]),
            "mask_score": float(score),
            "bbox_xyxy": bbox_global,
            "segmentation": rle,
        }

        all_annotations.append(annotation)
        all_boxes_global.append(bbox_global)
        all_masks_global.append(mask_global)
        all_labels.append(label)
        all_class_ids.append(det_counter)
        det_counter += 1

    return all_boxes_global, all_masks_global, all_labels, all_class_ids, all_annotations


def _infer_base_id(folder: Path) -> str:
    """Return patch base id (e.g. 'new' from compare_meta_new.json or patch_left_new_1.png)."""
    meta_files = sorted(folder.glob("compare_meta_*.json"))
    for p in meta_files:
        m = re.match(r"compare_meta_(.+)\.json$", p.name, re.IGNORECASE)
        if m:
            return m.group(1)

    for p in sorted(folder.glob("patch_left_*.png")):
        m = re.match(r"patch_left_(.+)_\d+\.png$", p.name, re.IGNORECASE)
        if m:
            return m.group(1)

    for p in sorted(folder.glob("compare_*.png")):
        m = re.match(r"compare_(.+)_\d+\.png$", p.name, re.IGNORECASE)
        if m:
            return m.group(1)

    raise ValueError(f"Could not infer patch base id from folder: {folder}")


def _load_compare_meta(folder: Path) -> tuple[Path, dict]:
    meta_files = sorted(folder.glob("compare_meta_*.json"))
    if not meta_files:
        raise FileNotFoundError(f"No compare_meta_*.json found in {folder}")
    if len(meta_files) > 1:
        raise RuntimeError(f"Multiple compare_meta_*.json files found: {meta_files}")
    meta_path = meta_files[0]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or not meta:
        raise ValueError(f"{meta_path} is empty or invalid")
    return meta_path, meta


def _find_new_pano(folder: Path, base_id: str) -> Path:
    """Locate the NEW panorama image for the given patch base id."""
    meta_path = folder / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for name in meta.keys():
            if Path(name).stem == base_id:
                candidate = folder / "images" / name
                if candidate.exists():
                    return candidate

    images_dir = folder / "images"
    if images_dir.is_dir():
        for p in sorted(images_dir.iterdir()):
            if p.is_file() and p.stem == base_id:
                return p

    raise FileNotFoundError(
        f"Could not find NEW panorama for base '{base_id}' under {folder / 'images'}"
    )



# ------------------ MAIN ------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Run Grounded-DINO + SAM2 segmentation on change-detection outputs."
    )
    parser.add_argument("--gs-root", default=DEFAULT_GSAM2_ROOT, help="Path to Grounded-SAM-2 repo.")
    parser.add_argument("--folder", default=DATA_DIR, help="Pipeline output folder (e.g. /data or data/run1).")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    gs_root = Path(args.gs_root).resolve()
    print(f"[Config] gs_root={gs_root}")
    print(f"[Config] folder={folder}")

    base_id = _infer_base_id(folder)
    _, compare_meta = _load_compare_meta(folder)
    pano_path = _find_new_pano(folder, base_id)

    batch = load_latest_batch_results(folder)
    pair_to_result = build_pair_to_result(batch)
    model_name = batch.get("model", "unknown").split("/")[-1]
    print(f"[Batch] loaded {model_name} results from {folder}")

    pano_bgr = cv2.imread(str(pano_path))
    if pano_bgr is None:
        raise SystemExit(f"Could not read pano: {pano_path}")
    pano_h, pano_w = pano_bgr.shape[:2]
    print(f"[Pano] {pano_path.name} size: {pano_w}x{pano_h}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Config] device={device}")
    sam2_predictor = build_sam2_predictor(gs_root=gs_root, device=device)
    gdino_model = build_gdino_model(gs_root=gs_root, device=device)

    global_boxes = []
    global_masks = []
    global_labels = []
    global_class_ids = []
    global_annotations = []

    items = sorted(compare_meta.items(), key=lambda kv: kv[1].get("index", 0))
    for view_key, entry in items:
        idx = entry.get("index")
        if idx is None:
            continue

        pair_name = f"{base_id}_{int(idx):02d}"
        qwen_result = pair_to_result.get(pair_name)
        if qwen_result is None:
            print(f"\n=== Skipping {view_key}: no batch result for pair {pair_name} ===")
            continue

        if qwen_result.get("no_change") or not qwen_result.get("changes"):
            print(f"\n=== Skipping {view_key}: no changes for pair {pair_name} ===")
            continue

        print(f"\n=== View idx={idx} key={view_key} pair={pair_name} ===")

        boxes_g, masks_g, labels_g, class_ids_g, ann_g = process_view_on_pano(
            idx=idx,
            view_key=view_key,
            meta_entry=entry,
            pano_bgr=pano_bgr,
            qwen_result=qwen_result,
            sam2_predictor=sam2_predictor,
            gdino_model=gdino_model,
            device=device,
            folder=folder,
            base_id=base_id,
        )

        global_boxes.extend(boxes_g)
        global_masks.extend(masks_g)
        global_labels.extend(labels_g)
        global_class_ids.extend(class_ids_g)
        global_annotations.extend(ann_g)

    if not global_boxes:
        print("No detections on pano.")
        return

    global_boxes = np.stack(global_boxes, axis=0)
    global_masks = np.stack(global_masks, axis=0)
    global_class_ids = np.arange(len(global_boxes), dtype=int)

    detections = sv.Detections(
        xyxy=global_boxes,
        mask=global_masks,
        class_id=global_class_ids,
    )

    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated = pano_bgr.copy()
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = mask_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=global_labels,
    )

    out_root = folder / f"gdsam2_{model_name}_outputs"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[Config] out_root={out_root}")

    pano_out_img = out_root / f"gd_sam2_{model_name}_annotated.jpg"
    cv2.imwrite(str(pano_out_img), annotated)
    print(f"[OUT] saved annotated pano → {pano_out_img}")

    union_mask = np.any(global_masks, axis=0)
    union_mask_u8 = (union_mask.astype("uint8") * 255)
    pano_mask_png = out_root / f"gd_sam2_{model_name}_union_mask.png"
    cv2.imwrite(str(pano_mask_png), union_mask_u8)
    print(f"[OUT] saved union mask → {pano_mask_png}")

    pano_masked = pano_bgr.copy().astype("float32")
    bg_factor = 0.3
    pano_masked[~union_mask] *= bg_factor
    purple = np.array([255, 0, 255], dtype="float32")
    alpha = 0.4
    pano_masked[union_mask] = (
        (1.0 - alpha) * pano_masked[union_mask] + alpha * purple
    )
    pano_masked = np.clip(pano_masked, 0, 255).astype("uint8")
    pano_masked_img = out_root / f"gd_sam2_{model_name}_masked.jpg"
    cv2.imwrite(str(pano_masked_img), pano_masked)
    print(f"[OUT] saved masked pano → {pano_masked_img}")

    annotations_path = out_root / f"gd_sam2_{model_name}_annotations.json"
    annotations_path.write_text(
        json.dumps(global_annotations, indent=2),
        encoding="utf-8",
    )
    print(f"[OUT] saved annotations → {annotations_path}")


if __name__ == "__main__":
    main()
