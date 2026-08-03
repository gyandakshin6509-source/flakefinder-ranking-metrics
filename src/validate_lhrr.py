"""
validate_lhrr.py — Run LHRR on three size-representative chip_0 flakes and
save 4-panel diagnostic figures to outputs/lhrr/.

Usage:
    python src/validate_lhrr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make src/ importable when running from project root
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lhrr import save_lhrr_figure

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
CHIP_DIR = DATA_ROOT / "_extracted" / "run_20260505_1616" / "chip_0"
FLATFIELD_NPY = DATA_ROOT / "flakes data" / "flakes data" / "flatfields" / "flatfield_10x_bin3.npy"
OUT_DIR = PROJECT_ROOT / "outputs" / "lhrr"

# ---------------------------------------------------------------------------
# Validation labels (chip_0, hBN run — size-based selection)
# ---------------------------------------------------------------------------

VALIDATION_FLAKES = [
    ("small",  "rank04_frame_0389_d28"),   # 505 µm²  (actual smallest)
    ("median", "rank13_frame_0149_d5"),    # 978 µm²  (index 10 of 21)
    ("large",  "rank02_frame_0257_d2"),    # 4556 µm² (actual largest)
]

MATERIAL = "hbn_medium"


def main() -> None:
    if not CHIP_DIR.exists():
        print(f"ERROR: chip_dir not found: {CHIP_DIR}")
        print("Run src/recon.py first to extract the archives.")
        sys.exit(1)

    if not FLATFIELD_NPY.exists():
        print(f"ERROR: flatfield .npy not found: {FLATFIELD_NPY}")
        sys.exit(1)

    flatfield = np.load(str(FLATFIELD_NPY)).astype(np.float32)
    print(f"Flatfield loaded: shape={flatfield.shape} dtype={flatfield.dtype}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for size_tag, label in VALIDATION_FLAKES:
        print(f"\n--- {size_tag}: {label} ---")
        out_path = OUT_DIR / f"chip_0_{label}.png"
        try:
            result = save_lhrr_figure(CHIP_DIR, label, flatfield, out_path,
                                      material=MATERIAL)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        skip = result["lhrr_skip_reason"]
        if skip:
            print(f"  Skipped: {skip}")
            print(f"  Figure: {out_path}")
            continue

        print(f"  lhrr_area_px  = {result['lhrr_area_px']}")
        print(f"  lhrr_area_um2 = {result['lhrr_area_um2']} µm²")
        print(f"  lhrr_fraction = {result['lhrr_fraction']}")
        print(f"  lhrr_bbox_frame = {result['lhrr_bbox_frame']}")
        print(f"  lhrr_bbox_stage = {result['lhrr_bbox_stage']}")
        print(f"  variance_threshold_used = {result['variance_threshold_used']}")
        print(f"  kernel_size_px = {result['kernel_size_px']}")
        print(f"  lhrr_quality_flag = {result['lhrr_quality_flag']}")
        print(f"  Figure saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
