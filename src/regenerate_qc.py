"""
regenerate_qc.py — rebuild every chip's occupancy map and save its QC PNG
under the run-prefixed naming convention. Used once after the chip_N collision
fix in flake_metrics.py.

Usage:
    python src/regenerate_qc.py
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eop import build_chip_occupancy, save_chip_occupancy_qc

PROJECT_ROOT = Path(__file__).parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "_extracted"
EOP_DIR = PROJECT_ROOT / "outputs" / "eop"
FLATFIELD_NPY = (
    PROJECT_ROOT / "data" / "flakes data" / "flakes data"
    / "flatfields" / "flatfield_10x_bin3.npy"
)

RUNS = ["run_20260505_1616", "run_20260504_1026"]


def main() -> None:
    flatfield = np.load(str(FLATFIELD_NPY)).astype(np.float32)
    EOP_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for run_name in RUNS:
        run_dir = EXTRACTED_DIR / run_name
        if not run_dir.exists():
            print(f"SKIP: {run_dir} not found")
            continue
        for chip_dir in sorted(p for p in run_dir.iterdir()
                               if p.is_dir() and p.name.startswith("chip_")):
            print(f"\n=== {run_name} / {chip_dir.name} ===")
            ts = time.time()
            cd = build_chip_occupancy(chip_dir, flatfield, progress_every=200)
            print(f"  occupancy build: {time.time() - ts:.1f} s")
            qc = EOP_DIR / f"{run_name}_{chip_dir.name}_occupancy.png"
            save_chip_occupancy_qc(
                cd, qc,
                title=f"{run_name} / {chip_dir.name} — {cd['n_detections']} detections "
                      f"(material baseline {cd['substrate_baseline']:.1f})",
            )
            print(f"  saved: {qc}")
            del cd
            gc.collect()
    print(f"\nTotal: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
