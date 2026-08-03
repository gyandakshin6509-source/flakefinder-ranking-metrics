"""
diag_eop_resolution_only.py: Verify the area-integral fix for
weighted_obstruction by re-running the resolution sensitivity test for the
three validation flakes at OCCUPANCY_STAGE_PX_UM = 5.0, 2.0, 1.0.

Usage:
    python src/diag_eop_resolution_only.py
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

import eop
from eop import build_chip_occupancy, compute_eop

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
CHIP_DIR = DATA_ROOT / "_extracted" / "run_20260505_1616" / "chip_0"
FLATFIELD_NPY = DATA_ROOT / "flakes data" / "flakes data" / "flatfields" / "flatfield_10x_bin3.npy"

VAL_FLAKES = [
    ("small",  "rank04_frame_0389_d28"),
    ("median", "rank13_frame_0149_d5"),
    ("large",  "rank02_frame_0257_d2"),
]

PX_SETTINGS = (5.0, 2.0, 1.0)


def main() -> None:
    if not CHIP_DIR.exists():
        print(f"ERROR: chip_dir not found: {CHIP_DIR}")
        sys.exit(1)
    flatfield = np.load(str(FLATFIELD_NPY)).astype(np.float32)

    rows: dict[str, dict[float, dict[str, float]]] = {label: {} for _, label in VAL_FLAKES}

    for px in PX_SETTINGS:
        print("=" * 78)
        print(f"Building chip_0 occupancy at OCCUPANCY_STAGE_PX_UM = {px}")
        print("=" * 78)
        eop.OCCUPANCY_STAGE_PX_UM = px
        t0 = time.time()
        cd = build_chip_occupancy(CHIP_DIR, flatfield, progress_every=200)
        print(f"  build time: {time.time() - t0:.1f} s")
        for _, label in VAL_FLAKES:
            r = compute_eop(CHIP_DIR, label, cd)
            rows[label][px] = {
                "clearance": float(r["clearance_um"]),
                "wobs": float(r["weighted_obstruction"]),
                "eop": float(r["eop_score"]),
            }
            print(f"  {label}: clearance={r['clearance_um']:.1f} µm  "
                  f"wobs={r['weighted_obstruction']:.4f}  eop={r['eop_score']:.2f}")
        del cd
        gc.collect()

    print("\n" + "=" * 78)
    print("Resolution sensitivity (after area-integral fix)")
    print("=" * 78)
    print(f"  {'flake':<28} "
          f"{'px=5.0 (cle | wobs | eop)':>32} "
          f"{'px=2.0 (cle | wobs | eop)':>32} "
          f"{'px=1.0 (cle | wobs | eop)':>32}")
    print("  " + "-" * 124)
    for _, label in VAL_FLAKES:
        cells = []
        for px in PX_SETTINGS:
            r = rows[label][px]
            cells.append(f"{r['clearance']:>5.1f} | {r['wobs']:>6.3f} | {r['eop']:>5.2f}")
        print(f"  {label:<28} {cells[0]:>32} {cells[1]:>32} {cells[2]:>32}")

    print("\n  EOP score variation across resolutions (max-min as % of max):")
    for _, label in VAL_FLAKES:
        eops = [rows[label][px]["eop"] for px in PX_SETTINGS]
        mx, mn = max(eops), min(eops)
        var_pct = 100.0 * (mx - mn) / mx if mx > 0 else 0.0
        verdict = "STABLE" if var_pct <= 5.0 else "UNSTABLE"
        print(f"    {label:<28} EOP range [{mn:.2f}, {mx:.2f}]  "
              f"variation = {var_pct:.1f}%  -> {verdict}")

    print("\nDone.")


if __name__ == "__main__":
    main()
