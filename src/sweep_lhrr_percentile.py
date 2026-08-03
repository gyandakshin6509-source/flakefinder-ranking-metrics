"""
sweep_lhrr_percentile.py: Sensitivity check on VARIANCE_PERCENTILE.

Runs LHRR at percentiles 25, 50, and 75 for the three chip_0 validation
flakes and saves 4-panel figures to outputs/lhrr_sweep/.
Prints a summary table of LHRR area and fraction at each percentile.

Usage:
    python src/sweep_lhrr_percentile.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lhrr import save_lhrr_figure

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
CHIP_DIR = DATA_ROOT / "_extracted" / "run_20260505_1616" / "chip_0"
FLATFIELD_NPY = DATA_ROOT / "flakes data" / "flakes data" / "flatfields" / "flatfield_10x_bin3.npy"
OUT_DIR = PROJECT_ROOT / "outputs" / "lhrr_sweep"

VALIDATION_FLAKES = [
    ("small",  "rank04_frame_0389_d28"),
    ("median", "rank13_frame_0149_d5"),
    ("large",  "rank02_frame_0257_d2"),
]

PERCENTILES = [25, 50, 75]
MATERIAL = "hbn_medium"


def main() -> None:
    if not CHIP_DIR.exists():
        print(f"ERROR: chip_dir not found: {CHIP_DIR}")
        sys.exit(1)

    flatfield = np.load(str(FLATFIELD_NPY)).astype(np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # rows: (size_tag, label, percentile) -> result
    rows: list[tuple[str, str, int, dict]] = []

    for size_tag, label in VALIDATION_FLAKES:
        for pct in PERCENTILES:
            out_path = OUT_DIR / f"{label}_p{pct:02d}.png"
            try:
                result = save_lhrr_figure(
                    CHIP_DIR, label, flatfield, out_path,
                    material=MATERIAL,
                    variance_percentile=pct,
                )
            except Exception as exc:
                print(f"FAILED {label} p{pct}: {exc}")
                result = {"lhrr_area_um2": None, "lhrr_fraction": None,
                          "lhrr_skip_reason": str(exc)}
            rows.append((size_tag, label, pct, result))
            skip = result.get("lhrr_skip_reason")
            status = f"skip={skip}" if skip else f"{out_path.name}"
            print(f"  {label}  p{pct:02d}  -> {status}")

    # Summary table
    print()
    print(f"{'Flake':<10} {'Label':<30} {'p25 area':>10} {'p25 frac':>9} "
          f"{'p50 area':>10} {'p50 frac':>9} {'p75 area':>10} {'p75 frac':>9}")
    print("-" * 103)

    for size_tag, label in VALIDATION_FLAKES:
        cells: dict[int, dict] = {}
        for st, lb, pct, res in rows:
            if lb == label:
                cells[pct] = res

        def fmt_area(r: dict) -> str:
            v = r.get("lhrr_area_um2")
            return f"{v:.0f} µm²" if v is not None else "skip"

        def fmt_frac(r: dict) -> str:
            v = r.get("lhrr_fraction")
            return f"{v:.3f}" if v is not None else "skip"

        print(
            f"{size_tag:<10} {label:<30} "
            f"{fmt_area(cells[25]):>10} {fmt_frac(cells[25]):>9} "
            f"{fmt_area(cells[50]):>10} {fmt_frac(cells[50]):>9} "
            f"{fmt_area(cells[75]):>10} {fmt_frac(cells[75]):>9}"
        )

    print()
    print(f"Figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
