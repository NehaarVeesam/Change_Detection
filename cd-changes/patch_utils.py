"""Patch discovery and model output parsing."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def discover_patch_pairs(folder: Path) -> List[Tuple[Path, Path, str, int]]:
    """
    Finds pairs like:
      patch_left_<base>_<idx>.png  (NEW)
      patch_right_<base>_<idx>.png (OLD)

    Returns list of (newA, oldB, base, idx)
    """
    folder = folder.resolve()
    lefts = (
        sorted(folder.glob("patch_left_*.png"))
        + sorted(folder.glob("patch_left_*.jpg"))
        + sorted(folder.glob("patch_left_*.jpeg"))
    )
    rights = (
        sorted(folder.glob("patch_right_*.png"))
        + sorted(folder.glob("patch_right_*.jpg"))
        + sorted(folder.glob("patch_right_*.jpeg"))
    )

    def parse(p: Path, side: str) -> Optional[Tuple[str, int]]:
        m = re.match(rf"patch_{side}_(.+)_(\d+)\.(png|jpg|jpeg)$", p.name, re.IGNORECASE)
        if not m:
            return None
        return m.group(1), int(m.group(2))

    left_map: Dict[Tuple[str, int], Path] = {}
    for p in lefts:
        k = parse(p, "left")
        if k:
            left_map[k] = p

    right_map: Dict[Tuple[str, int], Path] = {}
    for p in rights:
        k = parse(p, "right")
        if k:
            right_map[k] = p

    keys = sorted(set(left_map.keys()) & set(right_map.keys()))
    return [
        (left_map[(base, idx)], right_map[(base, idx)], base, idx)
        for (base, idx) in keys
    ]


def strip_thinking(text: str) -> str:
    """Remove Qwen thinking blocks before JSON extraction."""
    text = text.strip()
    think_open, think_close = "<" + "think" + ">", "</" + "think" + ">"
    for open_tag, close_tag in (
        (think_open, think_close),
        ("<think>", "</think>"),
    ):
        while open_tag in text:
            start = text.find(open_tag)
            end = text.find(close_tag, start)
            if end == -1:
                break
            text = (text[:start] + text[end + len(close_tag) :]).strip()
    return text.strip()


def json_from_text(text: str) -> Dict[str, Any]:
    """Robustly extract JSON object from model output."""
    text = strip_thinking(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {"comparable": False, "no_change": True, "changes": [], "raw_text": text}

    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"comparable": False, "no_change": True, "changes": [], "raw_text": text}


def normalize_and_filter(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize expected keys from model output."""
    out: Dict[str, Any] = {
        "comparable": bool(raw.get("comparable", False)),
        "overlap_score": float(raw.get("overlap_score", 0.0) or 0.0),
        "viewpoint_shift": raw.get("viewpoint_shift", "") or "",
        "image_a_description": raw.get("image_a_description", "") or "",
        "image_b_description": raw.get("image_b_description", "") or "",
    }
    changes = raw.get("changes", []) or []
    if isinstance(changes, dict):
        changes = [changes]
    normalized = []
    for c in changes:
        if not isinstance(c, dict):
            continue
        object_name = (c.get("object_name") or c.get("target_object") or "").strip()
        if not object_name:
            continue
        normalized.append(
            {
                "change_type": (c.get("change_type") or c.get("status") or "").strip(),
                "object_name": object_name,
                "object_names_alternatives": c.get("object_names_alternatives")
                or c.get("caption_alternatives")
                or [],
                "location": c.get("location", "") or "",
                "description": c.get("description", "") or c.get("evidence", "") or "",
                "before_state": c.get("before_state", "") or "",
                "after_state": c.get("after_state", "") or "",
                "confidence": float(c.get("confidence", 0.0) or 0.0),
            }
        )
    out["changes"] = normalized
    out["no_change"] = bool(raw.get("no_change", len(normalized) == 0))
    return out
