"""Precise variance distribution diagnostic for rank04_frame_0389_d28."""
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

LABEL = "rank04_frame_0389_d28"
PCT = 25

def odd_round(v):
    n = max(3, int(round(v)))
    return n if n % 2 == 1 else n + 1

def main():
    import re
    flatfield = np.load(str(FLATFIELD_NPY)).astype(np.float32)

    m = re.match(r"^rank(\d+)_frame_(\d+)_d(\d+)$", LABEL)
    _, frame_n, _ = int(m.group(1)), int(m.group(2)), int(m.group(3))

    det, frame_entry, meta = lookup_detection(CHIP_DIR, LABEL)
    W, H = get_frame_dims(meta)
    pixel_um = get_pixel_um(meta)

    raw = cv2.imread(str(CHIP_DIR / "scan_10x" / f"frame_{frame_n:04d}.jpg"))
    corrected = apply_flatfield(raw, flatfield)

    mask_full = build_flake_mask(det, H, W)
    kernel_px = odd_round(lhrr_mod.VARIANCE_KERNEL_UM / pixel_um)

    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_dilated = cv2.dilate(mask_full.astype(np.uint8), disk).astype(bool)

    bbox = det.get("bbox")
    bx, by, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    pad = max(lhrr_mod.CROP_PAD_PX, kernel_px + 2)
    row0 = max(0, by - pad); row1 = min(H, by + bh + pad)
    col0 = max(0, bx - pad); col1 = min(W, bx + bw + pad)

    green_crop  = corrected[row0:row1, col0:col1, 1].astype(np.float32)
    mask_crop   = mask_full[row0:row1, col0:col1]
    mask_d_crop = mask_dilated[row0:row1, col0:col1]
    buffer_crop = mask_d_crop & ~mask_crop

    k = kernel_px
    box_k = np.ones((k, k), dtype=np.float32) / (k * k)
    mean_map    = cv2.filter2D(green_crop, -1, box_k)
    mean_sq_map = cv2.filter2D(green_crop ** 2, -1, box_k)
    var_map = np.clip(mean_sq_map - mean_map ** 2, 0.0, None)

    v_mask   = var_map[mask_crop]
    v_dil    = var_map[mask_d_crop]
    v_buf    = var_map[buffer_crop]

    n_mask  = int(mask_crop.sum())
    n_dil   = int(mask_d_crop.sum())
    n_buf   = int(buffer_crop.sum())

    pct_mask = float(np.percentile(v_mask, PCT))
    pct_dil  = float(np.percentile(v_dil,  PCT))
    rel_diff = (pct_mask - pct_dil) / pct_dil * 100

    print(f"Label: {LABEL}  (kernel_px={kernel_px})")
    print()
    print(f"  Pixel counts")
    print(f"    mask (strict interior) : {n_mask:>6d} px")
    print(f"    mask_dilated           : {n_dil:>6d} px")
    print(f"    buffer (dil - strict)  : {n_buf:>6d} px  ({n_buf/n_dil*100:.1f}% of dilated)")
    print()
    print(f"  Mean variance")
    print(f"    var_map[mask]          : {v_mask.mean():>10.4f}")
    print(f"    var_map[mask_dilated]  : {v_dil.mean():>10.4f}")
    print(f"    var_map[buffer only]   : {v_buf.mean():>10.4f}")
    print()
    print(f"  25th percentile")
    print(f"    var_map[mask]          : {pct_mask:>10.4f}   <- thresh_after  (current code)")
    print(f"    var_map[mask_dilated]  : {pct_dil:>10.4f}   <- thresh_before")
    print(f"    absolute diff          : {pct_mask - pct_dil:>+10.4f}")
    print(f"    relative diff          : {rel_diff:>+9.2f}%")
    print()
    if abs(rel_diff) < 5:
        print("  VERDICT: <5% relative diff -> dilated-mask effect is empirically negligible for this flake.")
    elif abs(rel_diff) < 20:
        print("  VERDICT: 5-20% relative diff -> fix is real but modest.")
    else:
        print("  VERDICT: >20% relative diff -> fix is substantial.")

if __name__ == "__main__":
    main()
