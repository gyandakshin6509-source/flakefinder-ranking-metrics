"""Diagnostic: compare p25 threshold and clean-pixel count before/after dilated-mask fix."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2

SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from io_utils import apply_flatfield, build_flake_mask, get_frame_dims, get_pixel_um, lookup_detection
import lhrr as lhrr_mod

PROJECT_ROOT = Path(__file__).parent.parent
CHIP_DIR = PROJECT_ROOT / "data" / "_extracted" / "run_20260505_1616" / "chip_0"
FLATFIELD_NPY = PROJECT_ROOT / "data" / "flakes data" / "flakes data" / "flatfields" / "flatfield_10x_bin3.npy"

FLAKES = [
    ("small",  "rank04_frame_0389_d28"),
    ("median", "rank13_frame_0149_d5"),
    ("large",  "rank02_frame_0257_d2"),
]
PCT = 25
CROP_PAD_PX = lhrr_mod.CROP_PAD_PX
VARIANCE_KERNEL_UM = lhrr_mod.VARIANCE_KERNEL_UM

def odd_round(v):
    n = max(3, int(round(v)))
    return n if n % 2 == 1 else n + 1

def parse_label(label):
    import re
    m = re.match(r"^rank(\d+)_frame_(\d+)_d(\d+)$", label)
    return int(m.group(1)), int(m.group(2)), int(m.group(3))

def run_diag(label, flatfield):
    _, frame_n, _ = parse_label(label)
    det, frame_entry, meta = lookup_detection(CHIP_DIR, label)
    W, H = get_frame_dims(meta)
    pixel_um = get_pixel_um(meta)

    raw = cv2.imread(str(CHIP_DIR / "scan_10x" / f"frame_{frame_n:04d}.jpg"))
    corrected = apply_flatfield(raw, flatfield)

    mask_full = build_flake_mask(det, H, W)
    kernel_px = odd_round(VARIANCE_KERNEL_UM / pixel_um)

    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_dilated = cv2.dilate(mask_full.astype(np.uint8), disk).astype(bool)

    bbox = det.get("bbox")
    if bbox is None:
        ys, xs = np.where(mask_full)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]
    bx, by, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    pad = max(CROP_PAD_PX, kernel_px + 2)
    row0 = max(0, by - pad); row1 = min(H, by + bh + pad)
    col0 = max(0, bx - pad); col1 = min(W, bx + bw + pad)

    green_crop = corrected[row0:row1, col0:col1, 1].astype(np.float32)
    mask_crop   = mask_full[row0:row1, col0:col1]
    mask_d_crop = mask_dilated[row0:row1, col0:col1]

    k = kernel_px
    box_k = np.ones((k, k), dtype=np.float32) / (k * k)
    mean_map    = cv2.filter2D(green_crop, -1, box_k)
    mean_sq_map = cv2.filter2D(green_crop ** 2, -1, box_k)
    var_map = np.clip(mean_sq_map - mean_map ** 2, 0.0, None)

    # BEFORE: sample from dilated mask
    thresh_before = float(np.percentile(var_map[mask_d_crop], PCT))
    clean_before  = int(((var_map < thresh_before) & mask_crop).sum())

    # AFTER: sample from strict mask
    thresh_after  = float(np.percentile(var_map[mask_crop], PCT))
    clean_after   = int(((var_map < thresh_after) & mask_crop).sum())

    n_strict  = int(mask_crop.sum())
    n_dilated = int(mask_d_crop.sum())

    return {
        "thresh_before": thresh_before,
        "thresh_after":  thresh_after,
        "diff":          thresh_after - thresh_before,
        "clean_before":  clean_before,
        "clean_after":   clean_after,
        "clean_diff":    clean_after - clean_before,
        "n_strict_px":   n_strict,
        "n_dilated_px":  n_dilated,
        "buffer_px":     n_dilated - n_strict,
    }

def main():
    flatfield = np.load(str(FLATFIELD_NPY)).astype(np.float32)

    print(f"\n{'Flake':<10} {'Label':<30} "
          f"{'thresh_before':>14} {'thresh_after':>13} {'diff':>9} "
          f"{'clean_before':>13} {'clean_after':>12} {'clean_diff':>11} "
          f"{'strict_px':>10} {'dilated_px':>11} {'buffer_px':>10}")
    print("-" * 147)

    for size_tag, label in FLAKES:
        d = run_diag(label, flatfield)
        print(f"{size_tag:<10} {label:<30} "
              f"{d['thresh_before']:>14.4f} {d['thresh_after']:>13.4f} {d['diff']:>+9.4f} "
              f"{d['clean_before']:>13d} {d['clean_after']:>12d} {d['clean_diff']:>+11d} "
              f"{d['n_strict_px']:>10d} {d['n_dilated_px']:>11d} {d['buffer_px']:>10d}")

    print()
    print("Notes:")
    print("  thresh_before  = np.percentile(var_map[mask_dilated], 25)")
    print("  thresh_after   = np.percentile(var_map[mask_crop],    25)  <- current code")
    print("  clean_*        = count of pixels where (var_map < threshold) & mask_crop")
    print("  buffer_px      = extra pixels in dilated sample vs strict sample")

if __name__ == "__main__":
    main()
