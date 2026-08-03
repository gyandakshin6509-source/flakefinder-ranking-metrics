"""
validate_eop.py: Build chip_0 (hBN) occupancy + obstruction maps and run EOP
on the three size-based validation flakes.

Usage:
    python src/validate_eop.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eop import (
    build_chip_occupancy,
    compute_eop,
    save_chip_occupancy_qc,
    save_eop_figure,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
CHIP_DIR = DATA_ROOT / "_extracted" / "run_20260505_1616" / "chip_0"
FLATFIELD_NPY = DATA_ROOT / "flakes data" / "flakes data" / "flatfields" / "flatfield_10x_bin3.npy"
OUT_DIR = PROJECT_ROOT / "outputs" / "eop"

VALIDATION_FLAKES = [
    ("small",  "rank04_frame_0389_d28"),
    ("median", "rank13_frame_0149_d5"),
    ("large",  "rank02_frame_0257_d2"),
]


def main() -> None:
    if not CHIP_DIR.exists():
        print(f"ERROR: chip_dir not found: {CHIP_DIR}")
        sys.exit(1)
    if not FLATFIELD_NPY.exists():
        print(f"ERROR: flatfield not found: {FLATFIELD_NPY}")
        sys.exit(1)

    flatfield = np.load(str(FLATFIELD_NPY)).astype(np.float32)
    print(f"Flatfield loaded: shape={flatfield.shape}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Building chip_0 (hBN) occupancy + obstruction maps ===")
    t0 = time.time()
    chip_data = build_chip_occupancy(CHIP_DIR, flatfield, progress_every=50)
    elapsed = time.time() - t0
    print(f"  build time: {elapsed:.1f} s")

    qc_path = OUT_DIR / "chip_0_occupancy_qc.png"
    save_chip_occupancy_qc(chip_data, qc_path,
                           title=f"chip_0 hBN: {chip_data['n_detections']} detections")
    print(f"  QC image: {qc_path}")

    print(f"\n=== Per-flake EOP ===")
    for size_tag, label in VALIDATION_FLAKES:
        print(f"\n--- {size_tag}: {label} ---")
        try:
            result = compute_eop(CHIP_DIR, label, chip_data)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        if result["eop_skip_reason"]:
            print(f"  SKIPPED: {result['eop_skip_reason']}")
            continue

        out_path = OUT_DIR / f"chip_0_{label}.png"
        save_eop_figure(CHIP_DIR, label, chip_data, result, out_path)

        print(f"  clearance_um         = {result['clearance_um']}")
        print(f"  weighted_obstruction = {result['weighted_obstruction']}")
        print(f"  eop_score            = {result['eop_score']}")
        print(f"  clearance_warning    = {result['clearance_warning']}")
        print(f"  cand_pixels (map)    = {result['cand_pixels']}")
        print(f"  stage_center_um      = {result['stage_center_um']}")
        print(f"  Figure: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
