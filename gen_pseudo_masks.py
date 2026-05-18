"""
gen_pseudo_masks.py
-------------------
Generate SAM-2.1 pseudo-masks for the HIT-UAV and DroneVehicle datasets, using
the ground-truth bounding boxes as box prompts. The generated masks are then
used as auxiliary segmentation supervision (BCE only) during training of the
IR-DKFL-UAV detector. This procedure follows the pseudo-label strategy of YOLOE
(Wang et al., 2025).

Pipeline
--------
1.  Load each image and its YOLO-format annotations (cls, xc, yc, w, h).
2.  Convert each normalized bbox to absolute pixel coordinates and feed it to
    SAM-2.1 as a single box prompt.
3.  Apply a 3x3 Gaussian smoothing kernel to the predicted binary mask.
4.  Discard masks whose total area is below the minimum-area threshold
    (default 16 pixels) to suppress noisy or degenerate predictions.
5.  Write the surviving masks to a per-image .npz file containing:
        - boxes:  (N, 4) absolute xyxy pixel boxes
        - cls:    (N,)   class ids (int)
        - masks:  (N, H, W) uint8 binary masks
        - kept:   (N,)   indices of source bboxes that produced valid masks

Output layout
-------------
    pseudo_masks/
        train/
            000001.npz
            000002.npz
            ...
        val/
            ...
        test/
            ...

Usage
-----
    # Requires the official SAM-2 repo and a downloaded SAM-2.1 checkpoint.
    pip install git+https://github.com/facebookresearch/segment-anything-2.git
    # Download: sam2.1_hiera_large.pt

    python gen_pseudo_masks.py \
        --images   datasets/HIT-UAV/images \
        --labels   datasets/HIT-UAV/labels \
        --output   datasets/HIT-UAV/pseudo_masks \
        --splits   train val test \
        --sam-cfg  sam2.1_hiera_l.yaml \
        --sam-ckpt checkpoints/sam2.1_hiera_large.pt \
        --device   cuda \
        --min-area 16 \
        --gaussian-ksize 3

Notes
-----
* Pseudo-mask generation is a one-time pre-processing step. SAM-2.1 is NOT
  invoked during model training or inference.
* For DroneVehicle (RGB-IR pairs), this script operates on the IR images only;
  the RGB modality is not used for our framework's pseudo-mask supervision.
* The script is idempotent: existing .npz files are skipped unless --overwrite
  is passed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as exc:
    raise SystemExit(
        "SAM-2 is not installed. Install via:\n"
        "    pip install git+https://github.com/facebookresearch/segment-anything-2.git"
    ) from exc


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ----------------------------------------------------------------------------- 
# I/O helpers
# -----------------------------------------------------------------------------
def read_image(path: Path) -> np.ndarray:
    """Read an image as 3-channel uint8 RGB. Single-channel IR is replicated."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_yolo_labels(path: Path, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a YOLO-format label file. Returns:
        boxes : (N, 4) float32 absolute xyxy
        cls   : (N,)   int32 class ids
    Empty if the file is missing or has zero rows.
    """
    if not path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            c, xc, yc, w, h = parts[:5]
            rows.append((int(c), float(xc), float(yc), float(w), float(h)))

    if not rows:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    arr = np.asarray(rows, dtype=np.float32)
    cls = arr[:, 0].astype(np.int32)
    xc, yc, w, h = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h

    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    # Clip to image
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, img_w - 1)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, img_h - 1)
    return boxes, cls


# -----------------------------------------------------------------------------
# Mask post-processing
# -----------------------------------------------------------------------------
def smooth_and_filter(
    mask: np.ndarray,
    gaussian_ksize: int = 3,
    min_area: int = 16,
) -> np.ndarray | None:
    """
    Apply a Gaussian smoothing kernel to the predicted soft mask and threshold
    it back to binary. Returns None if the resulting binary mask has fewer
    than `min_area` foreground pixels.
    """
    if mask.dtype != np.float32:
        mask = mask.astype(np.float32)

    if gaussian_ksize and gaussian_ksize > 1:
        # Kernel size must be odd
        k = gaussian_ksize if gaussian_ksize % 2 == 1 else gaussian_ksize + 1
        mask = cv2.GaussianBlur(mask, (k, k), sigmaX=0)

    binary = (mask >= 0.5).astype(np.uint8)
    if binary.sum() < min_area:
        return None
    return binary


# -----------------------------------------------------------------------------
# SAM-2.1 wrapper
# -----------------------------------------------------------------------------
class SAM21BoxPredictor:
    """Lightweight wrapper around SAM-2.1's image predictor for box prompts."""

    def __init__(self, cfg: str, ckpt: str, device: str = "cuda"):
        self.device = device
        self.model = build_sam2(cfg, ckpt, device=device)
        self.predictor = SAM2ImagePredictor(self.model)

    def predict(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        image : (H, W, 3) uint8 RGB
        boxes : (N, 4) float32 absolute xyxy
        returns: (N, H, W) float32 soft masks in [0, 1]
        """
        if boxes.shape[0] == 0:
            return np.zeros((0, image.shape[0], image.shape[1]), dtype=np.float32)

        self.predictor.set_image(image)
        masks, _scores, _logits = self.predictor.predict(
            box=boxes,
            multimask_output=False,
        )
        # SAM-2 returns (N, 1, H, W) or (N, H, W); normalize to (N, H, W)
        if masks.ndim == 4:
            masks = masks[:, 0]
        return masks.astype(np.float32)


# -----------------------------------------------------------------------------
# Per-image pipeline
# -----------------------------------------------------------------------------
def process_image(
    image_path: Path,
    label_path: Path,
    out_path: Path,
    predictor: SAM21BoxPredictor,
    gaussian_ksize: int,
    min_area: int,
    overwrite: bool,
) -> Tuple[int, int]:
    """Returns (n_boxes, n_kept)."""
    if out_path.exists() and not overwrite:
        return (0, 0)

    image = read_image(image_path)
    h, w = image.shape[:2]
    boxes, cls = read_yolo_labels(label_path, w, h)

    if boxes.shape[0] == 0:
        # Save an empty record so downstream code can index by image name
        np.savez_compressed(
            out_path,
            boxes=np.zeros((0, 4), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.int32),
            masks=np.zeros((0, h, w), dtype=np.uint8),
            kept=np.zeros((0,), dtype=np.int32),
            image_shape=np.asarray([h, w], dtype=np.int32),
        )
        return (0, 0)

    soft = predictor.predict(image, boxes)  # (N, H, W)

    kept_masks: List[np.ndarray] = []
    kept_idx: List[int] = []
    for i in range(soft.shape[0]):
        binary = smooth_and_filter(soft[i], gaussian_ksize=gaussian_ksize, min_area=min_area)
        if binary is not None:
            kept_masks.append(binary)
            kept_idx.append(i)

    if kept_masks:
        masks_arr = np.stack(kept_masks, axis=0).astype(np.uint8)
    else:
        masks_arr = np.zeros((0, h, w), dtype=np.uint8)
    kept_arr = np.asarray(kept_idx, dtype=np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        boxes=boxes[kept_arr] if kept_arr.size else np.zeros((0, 4), dtype=np.float32),
        cls=cls[kept_arr] if kept_arr.size else np.zeros((0,), dtype=np.int32),
        masks=masks_arr,
        kept=kept_arr,
        image_shape=np.asarray([h, w], dtype=np.int32),
    )
    return (int(boxes.shape[0]), int(kept_arr.size))


# -----------------------------------------------------------------------------
# Split orchestration
# -----------------------------------------------------------------------------
def gather_split(images_root: Path, labels_root: Path, split: str) -> List[Tuple[Path, Path]]:
    """Pair each image with its YOLO label file inside a split subdirectory."""
    img_dir = images_root / split
    lbl_dir = labels_root / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Split image directory not found: {img_dir}")

    pairs: List[Tuple[Path, Path]] = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        pairs.append((img_path, lbl_path))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate SAM-2.1 pseudo-masks for IR-DKFL-UAV training."
    )
    parser.add_argument("--images", required=True, type=Path,
                        help="Root images directory (contains <split>/ subdirs).")
    parser.add_argument("--labels", required=True, type=Path,
                        help="Root labels directory (YOLO .txt, contains <split>/ subdirs).")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output root directory for .npz pseudo-mask files.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Splits to process.")
    parser.add_argument("--sam-cfg",  required=True, type=str,
                        help="SAM-2.1 config yaml (e.g. sam2.1_hiera_l.yaml).")
    parser.add_argument("--sam-ckpt", required=True, type=str,
                        help="SAM-2.1 checkpoint path (e.g. sam2.1_hiera_large.pt).")
    parser.add_argument("--device", default="cuda", type=str,
                        help="Torch device (cuda | cpu).")
    parser.add_argument("--min-area", type=int, default=16,
                        help="Minimum mask area in pixels (default: 16).")
    parser.add_argument("--gaussian-ksize", type=int, default=3,
                        help="Gaussian smoothing kernel size, odd integer (default: 3).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing .npz files.")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        print("[WARN] CUDA not available, falling back to CPU. This will be slow.")
        args.device = "cpu"

    print(f"[INFO] Loading SAM-2.1: cfg={args.sam_cfg}, ckpt={args.sam_ckpt}")
    predictor = SAM21BoxPredictor(cfg=args.sam_cfg, ckpt=args.sam_ckpt, device=args.device)

    total_boxes = 0
    total_kept = 0
    t0 = time.time()

    for split in args.splits:
        pairs = gather_split(args.images, args.labels, split)
        print(f"[INFO] Split '{split}': {len(pairs)} images")
        out_split = args.output / split

        for img_path, lbl_path in tqdm(pairs, desc=f"  {split}", unit="img"):
            out_path = out_split / (img_path.stem + ".npz")
            n_box, n_kept = process_image(
                img_path, lbl_path, out_path, predictor,
                gaussian_ksize=args.gaussian_ksize,
                min_area=args.min_area,
                overwrite=args.overwrite,
            )
            total_boxes += n_box
            total_kept += n_kept

    elapsed = time.time() - t0
    drop = total_boxes - total_kept
    drop_pct = (drop / total_boxes * 100.0) if total_boxes else 0.0
    print(
        f"[DONE] Boxes={total_boxes}, kept={total_kept}, "
        f"dropped={drop} ({drop_pct:.2f}%) in {elapsed:.1f}s."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
