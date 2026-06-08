"""
Compare two equirectangular panoramas (new vs old) with a Three.js viewer.

Inputs:
  - --new-image, --old-image: panorama image paths
  - --metadata: JSON with rotation and position for each image
  - --output-dir: where staged assets, screenshots, and patches are written

Metadata JSON format:
{
  "new": { "rotation": [rx, ry, rz], "position": [x, y] or [x, y, z] },
  "old": { "rotation": [rx, ry, rz], "position": [x, y] or [x, y, z] }
}

Author(s):
Nehaar Veesam
"""

import argparse
import json
import os
import shutil
import socket
import threading
import time
from contextlib import closing
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urljoin

import cv2
import numpy as np
from playwright.sync_api import sync_playwright


def load_pair_metadata(metadata_path: Path) -> dict:
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    for side in ("new", "old"):
        if side not in data:
            raise ValueError(f"{metadata_path}: missing '{side}' entry")
        entry = data[side]
        rot = entry.get("rotation")
        pos = entry.get("position")
        if not rot or len(rot) < 3:
            raise ValueError(f"{metadata_path}: '{side}' needs 'rotation' as [rx, ry, rz]")
        if not pos or len(pos) < 2:
            raise ValueError(f"{metadata_path}: '{side}' needs 'position' as [x, y] or [x, y, z]")
    return data


def prepare_workspace(
    output_dir: Path,
    new_image: Path,
    old_image: Path,
    metadata: dict,
) -> tuple[Path, str, str]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    new_name = new_image.name
    old_name = old_image.name

    shutil.copy2(new_image, images_dir / new_name)
    shutil.copy2(old_image, images_dir / old_name)

    meta = {
        new_name: {
            "rotation": metadata["new"]["rotation"],
            "position": metadata["new"]["position"],
            "path": f"data_out/images/{new_name}",
        },
        old_name: {
            "rotation": metadata["old"]["rotation"],
            "position": metadata["old"]["position"],
            "path": f"data_out/images/{old_name}",
        },
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path, new_name, old_name


def find_free_port(preferred=8080):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        if preferred:
            try:
                s.bind(("", preferred))
                return preferred
            except OSError:
                pass
        s.bind(("", 0))
        return s.getsockname()[1]


def compute_patch_bounds(corners_left: dict, pano_w: int, pano_h: int):
    """
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


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs):
        pass


def serve(root: Path, port: int):
    os.chdir(str(root))
    server = ThreadingHTTPServer(("127.0.0.1", port), Quiet)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, t


def main():
    start_time = time.time()

    ap = argparse.ArgumentParser(
        description="Compare two panoramas (new vs old) and extract aligned patches."
    )
    ap.add_argument("--new-image", required=True, help="Path to the NEW panorama image")
    ap.add_argument("--old-image", required=True, help="Path to the OLD panorama image")
    ap.add_argument(
        "--metadata",
        required=True,
        help="JSON file with rotation and position for 'new' and 'old'",
    )
    ap.add_argument("--output-dir", default="/data", help="Output directory")

    ap.add_argument("--root", default="/app", help="Folder served by HTTP server")
    ap.add_argument("--page", default="split.html", help="HTML filename relative to --root")
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--count", type=int, default=6, help="Number of screenshots")
    ap.add_argument("--step", type=float, default=-60.0, help="Yaw step (deg) between shots")
    ap.add_argument("--port", type=int, default=8080)

    args = ap.parse_args()

    new_image = Path(args.new_image).resolve()
    old_image = Path(args.old_image).resolve()
    metadata_path = Path(args.metadata).resolve()
    output_dir = Path(args.output_dir).resolve()

    for path, label in (
        (new_image, "--new-image"),
        (old_image, "--old-image"),
        (metadata_path, "--metadata"),
    ):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_pair_metadata(metadata_path)
    meta_path, new_name, old_name = prepare_workspace(
        output_dir, new_image, old_image, metadata
    )

    print(f"width = {args.width}, height = {args.height}")
    print(f"count = {args.count}, step = {args.step}")
    print(f"new = {new_name}, old = {old_name}")

    root = Path(args.root).resolve()

    link = root / "data_out"
    if link.exists() or link.is_symlink():
        if link.is_symlink() and link.resolve() == output_dir:
            pass
        else:
            if link.is_symlink() or link.is_file():
                link.unlink()
            else:
                shutil.rmtree(link)
            link.symlink_to(output_dir, target_is_directory=True)
    else:
        link.symlink_to(output_dir, target_is_directory=True)

    page_path = (root / args.page).resolve()
    if not page_path.exists():
        raise SystemExit(f"Page not found: {page_path}")

    port = find_free_port(args.port)
    base = f"http://127.0.0.1:{port}/"
    server, thread = serve(root, port)
    url = urljoin(base, args.page)
    print(f"\nServing {root} at {url}")

    def to_meta_url(p: Path) -> str:
        root_abs = Path(args.root).resolve()
        p_abs = p if p.is_absolute() else (root_abs / p)
        try:
            rel = p_abs.relative_to(root_abs)
        except ValueError:
            raise SystemExit(
                f"ERROR: {p_abs} is not under --root {root_abs}. "
                "Place --output-dir inside --root or expose it under root (symlink)."
            )
        return urljoin(base, str(rel).replace("\\", "/"))

    meta_url = to_meta_url(link / "meta.json")

    new_src = output_dir / "images" / new_name
    old_src = output_dir / "images" / old_name
    new_pano_bgr = cv2.imread(str(new_src))
    old_pano_bgr = cv2.imread(str(old_src))
    if new_pano_bgr is None or old_pano_bgr is None:
        raise SystemExit(f"Failed to read images: {new_src}, {old_src}")

    print(f"\n=== Comparing {new_name} ↔ {old_name} ===")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(
                viewport={"width": args.width, "height": args.height},
                device_scale_factor=1,
            )
            page = ctx.new_page()
            page.on("console", lambda m: print("[console]", m.type, m.text))
            page.on("pageerror", lambda e: print("[pageerror]", e))

            page.goto(url, wait_until="load")
            page.wait_for_function(
                "() => window.viewer && typeof window.viewer.loadByNameFromSources === 'function'"
            )

            payload = {
                "left": {"name": new_name, "metaUrl": meta_url},
                "right": {"name": old_name, "metaUrl": meta_url},
            }
            page.evaluate(
                "async (p) => { await window.viewer.loadByNameFromSources(p); }",
                payload,
            )
            time.sleep(0.5)

            compare_meta = {}
            for i in range(1, args.count + 1):
                page.evaluate("(step) => window.viewer.rotateYaw(step)", args.step)
                time.sleep(0.05)

                filename = f"compare_{Path(new_name).stem}_{i}.png"
                out_file = output_dir / filename
                page.screenshot(path=str(out_file))
                print(f"Saved {out_file}")

                meta = page.evaluate(
                    "() => (window.viewer && typeof window.viewer.getCornerUVs === 'function') "
                    "? window.viewer.getCornerUVs() : null"
                )

                if meta is None:
                    print(f"[meta] WARNING: getCornerUVs() returned null/undefined for {filename}")
                    continue

                compare_meta[filename] = {
                    "index": i,
                    "file": filename,
                    "yaw_deg": args.step * i,
                    "step_deg": args.step,
                    "screenshot_width": args.width,
                    "screenshot_height": args.height,
                    "corners": meta,
                }

                corners_left = compare_meta[filename]["corners"]["left"]
                corners_right = compare_meta[filename]["corners"]["right"]
                new_patch_bgr, _ = extract_patch_from_pano(new_pano_bgr, corners_left)
                old_patch_bgr, _ = extract_patch_from_pano(old_pano_bgr, corners_right)

                patch_out_left = output_dir / f"patch_left_{Path(new_name).stem}_{i}.png"
                patch_out_right = output_dir / f"patch_right_{Path(new_name).stem}_{i}.png"
                cv2.imwrite(str(patch_out_left), new_patch_bgr)
                cv2.imwrite(str(patch_out_right), old_patch_bgr)

            if compare_meta:
                meta_out = output_dir / f"compare_meta_{Path(new_name).stem}.json"
                with open(meta_out, "w", encoding="utf-8") as f:
                    json.dump(compare_meta, f, indent=2)
                print(f"[meta] Saved {meta_out}")
            else:
                print("[meta] No metadata captured; meta json not written.")

            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        print("Done.")

    elapsed = time.time() - start_time
    print(f"Total time taken: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
