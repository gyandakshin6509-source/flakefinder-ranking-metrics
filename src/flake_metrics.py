"""
flake_metrics.py: runnable entrypoint for the FlakeFinder ranking-metrics
extension.

Computes LHRR + EOP for every revisit_50x candidate on every chip of one run.
Writes per-chip metrics JSONs and a run-level ranked_candidates.csv (sorted
descending by composite score). Prints per-run summary statistics.

Usage:
    python src/flake_metrics.py <path/to/run_dir>

  run_dir is the extracted root of one run archive
  (e.g. data/_extracted/run_20260505_1616).

# Composite ranking
# composite = COMPOSITE_W_LHRR * (lhrr_area_um2 / max(lhrr_area_um2 in run))
#           + COMPOSITE_W_EOP  * (eop_score    / max(eop_score    in run))
# Both terms are normalised by run-wide maxima after the chip loop, so
# composite scores are run-relative (best candidate scores near 1.0).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eop import build_chip_occupancy, compute_eop, save_chip_occupancy_qc
from lhrr import compute_lhrr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent

# Default only. Pass --flatfield to point at a different microscope config
# without touching this file.
FLATFIELD_PATH = (
    PROJECT_ROOT / "data" / "flakes data" / "flakes data"
    / "flatfields" / "flatfield_10x_bin3.npy"
)
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
EOP_DIR = OUTPUTS_DIR / "eop"

COMPOSITE_W_LHRR: float = 0.5
COMPOSITE_W_EOP: float = 0.5


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _null_lhrr(skip_reason: str) -> dict[str, Any]:
    return {
        "lhrr_area_px": None,
        "lhrr_area_um2": None,
        "lhrr_fraction": None,
        "lhrr_bbox_frame": None,
        "lhrr_bbox_stage": None,
        "variance_threshold_used": None,
        "kernel_size_px": None,
        "lhrr_quality_flag": None,
        "lhrr_skip_reason": skip_reason,
    }


def _null_eop(skip_reason: str) -> dict[str, Any]:
    return {
        "clearance_um": None,
        "weighted_obstruction": None,
        "eop_score": None,
        "clearance_warning": None,
        "cand_pixels": None,
        "stage_center_um": None,
        "eop_skip_reason": skip_reason,
    }


def _process_chip(
    chip_dir: Path,
    flatfield: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Process one chip. Returns (chip_meta, list of per-candidate records).
    """
    summary_path = chip_dir / "seg" / "summary.json"
    revisit_path = chip_dir / "seg" / "revisit_50x.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing seg/summary.json: {summary_path}")
    if not revisit_path.exists():
        raise FileNotFoundError(f"Missing seg/revisit_50x.json: {revisit_path}")

    summary = json.loads(summary_path.read_text())
    material = summary["params"]["material"]
    revisit = json.loads(revisit_path.read_text())
    points = revisit["points"]

    print(f"\n=== {chip_dir.name} (material={material}, "
          f"n_candidates={len(points)}) ===")

    print("  building chip occupancy + obstruction maps ...")
    t0 = time.time()
    chip_data = build_chip_occupancy(chip_dir, flatfield, progress_every=200)
    print(f"  occupancy build: {time.time() - t0:.1f} s")

    EOP_DIR.mkdir(parents=True, exist_ok=True)
    run_name = chip_dir.parent.name
    qc_path = EOP_DIR / f"{run_name}_{chip_dir.name}_occupancy.png"
    save_chip_occupancy_qc(
        chip_data,
        qc_path,
        title=f"{chip_dir.name} {material}: {chip_data['n_detections']} detections",
    )
    print(f"  occupancy QC -> {qc_path}")

    chip_records: list[dict[str, Any]] = []
    for pt in points:
        label = pt["label"]
        try:
            lhrr_res = compute_lhrr(chip_dir, label, flatfield, material=material)
        except Exception as exc:
            lhrr_res = _null_lhrr(f"exception: {type(exc).__name__}: {exc}")
        try:
            eop_res = compute_eop(chip_dir, label, chip_data)
        except Exception as exc:
            eop_res = _null_eop(f"exception: {type(exc).__name__}: {exc}")

        record: dict[str, Any] = {
            "chip": chip_dir.name,
            "material": material,
            "label": label,
            "stage_x_um": pt.get("x"),
            "stage_y_um": pt.get("y"),
            "stage_z_um": pt.get("z"),
            **lhrr_res,
            **eop_res,
        }
        chip_records.append(record)

        a = lhrr_res.get("lhrr_area_um2")
        e = eop_res.get("eop_score")
        a_str = f"{a:.0f} µm²" if a is not None else "-"
        e_str = f"{e:.1f}" if e is not None else "-"
        flag = lhrr_res.get("lhrr_quality_flag") or "?"
        warn = eop_res.get("clearance_warning")
        warn_str = "WARN" if warn else ("ok" if warn is False else "-")
        print(f"  {label:<28} lhrr={a_str:>11} ({flag:<8}) eop={e_str:>6} {warn_str}")

    chip_meta = {
        "chip": chip_dir.name,
        "material": material,
        "n_candidates": len(points),
        "occupancy": {
            "n_detections": chip_data["n_detections"],
            "substrate_baseline": chip_data["substrate_baseline"],
            "baseline_fallback": bool(chip_data["baseline_fallback"]),
            "max_raw_contrast": chip_data["max_raw_contrast"],
            "stage_px_um": chip_data["stage_px"],
            "map_origin_um": [
                float(chip_data["map_origin"][0]),
                float(chip_data["map_origin"][1]),
            ],
        },
    }
    return chip_meta, chip_records


def _normalise_and_rank(records: list[dict[str, Any]]) -> None:
    """Add lhrr_area_norm, eop_score_norm, composite_score in place."""
    areas = [r["lhrr_area_um2"] for r in records if r.get("lhrr_area_um2") is not None]
    eops = [r["eop_score"] for r in records if r.get("eop_score") is not None]
    max_area = max(areas) if areas else 0.0
    max_eop = max(eops) if eops else 0.0

    for r in records:
        a = r.get("lhrr_area_um2")
        e = r.get("eop_score")
        r["lhrr_area_norm"] = round(a / max_area, 6) if (a is not None and max_area > 0) else 0.0
        r["eop_score_norm"] = round(e / max_eop, 6) if (e is not None and max_eop > 0) else 0.0
        r["composite_score"] = round(
            COMPOSITE_W_LHRR * r["lhrr_area_norm"]
            + COMPOSITE_W_EOP * r["eop_score_norm"],
            6,
        )


CSV_COLUMNS = [
    "rank_in_run",
    "chip",
    "material",
    "label",
    "stage_x_um",
    "stage_y_um",
    "stage_z_um",
    "lhrr_area_um2",
    "lhrr_fraction",
    "lhrr_quality_flag",
    "lhrr_skip_reason",
    "clearance_um",
    "weighted_obstruction",
    "eop_score",
    "clearance_warning",
    "eop_skip_reason",
    "lhrr_area_norm",
    "eop_score_norm",
    "composite_score",
]


def _write_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(records, start=1):
            row = {**r, "rank_in_run": i}
            w.writerow(row)


def _print_run_summary(run_name: str, records: list[dict[str, Any]]) -> None:
    n = len(records)
    print("\n" + "=" * 78)
    print(f"Run summary: {run_name}")
    print("=" * 78)
    print(f"  candidates processed   : {n}")
    if n == 0:
        return

    by_material: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_material.setdefault(r["material"], []).append(r)

    n_warn = sum(1 for r in records if r.get("clearance_warning") is True)
    n_warn_none = sum(1 for r in records if r.get("clearance_warning") is None)
    n_lhrr_usable = sum(1 for r in records if r.get("lhrr_quality_flag") == "usable")
    n_lhrr_marginal = sum(1 for r in records if r.get("lhrr_quality_flag") == "marginal")
    n_lhrr_none = sum(1 for r in records if r.get("lhrr_quality_flag") is None)

    print(f"  clearance_warning=True : {n_warn}")
    print(f"  clearance_warning=False: {n - n_warn - n_warn_none}")
    print(f"  clearance_warning=None : {n_warn_none}")
    print(f"  lhrr_quality usable    : {n_lhrr_usable}")
    print(f"  lhrr_quality marginal  : {n_lhrr_marginal}")
    print(f"  lhrr_quality None      : {n_lhrr_none}")

    print(f"\n  composite score by material:")
    print(f"    {'material':<24} {'n':>4} {'mean':>8} {'median':>8} {'max':>8}")
    for mat, rs in sorted(by_material.items()):
        comps = [r["composite_score"] for r in rs]
        print(f"    {mat:<24} {len(rs):>4} "
              f"{statistics.mean(comps):>8.4f} "
              f"{statistics.median(comps):>8.4f} "
              f"{max(comps):>8.4f}")

    print(f"\n  top 5 by composite:")
    for r in records[:5]:
        a = r.get("lhrr_area_um2")
        e = r.get("eop_score")
        a_str = f"{a:.0f}" if a is not None else "-"
        e_str = f"{e:.1f}" if e is not None else "-"
        print(f"    {r.get('rank_in_run', '?'):>2} {r['chip']:<8} {r['label']:<28} "
              f"lhrr={a_str:>6} eop={e_str:>6} composite={r['composite_score']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run_dir: Path, flatfield_path: Path | None = None) -> None:
    flatfield_path = flatfield_path or FLATFIELD_PATH
    if not run_dir.exists():
        print(f"ERROR: run_dir not found: {run_dir}")
        sys.exit(1)
    if not flatfield_path.exists():
        print(f"ERROR: flatfield not found: {flatfield_path}")
        sys.exit(1)

    flatfield = np.load(str(flatfield_path)).astype(np.float32)
    print(f"Flatfield loaded: shape={flatfield.shape}")

    chip_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("chip_"))
    if not chip_dirs:
        print(f"ERROR: no chip_N directories in {run_dir}")
        sys.exit(1)
    print(f"Run: {run_dir.name}  chips: {[c.name for c in chip_dirs]}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    EOP_DIR.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    chip_metas: list[dict[str, Any]] = []
    t_run = time.time()

    for chip_dir in chip_dirs:
        chip_meta, chip_records = _process_chip(chip_dir, flatfield)
        chip_metas.append(chip_meta)
        chip_path = METRICS_DIR / f"{run_dir.name}_{chip_dir.name}_metrics.json"
        chip_payload = {
            **chip_meta,
            "candidates": chip_records,
        }
        with chip_path.open("w", encoding="utf-8") as f:
            json.dump(chip_payload, f, indent=2, default=_json_default)
        print(f"  per-chip JSON -> {chip_path}")
        all_records.extend(chip_records)

    _normalise_and_rank(all_records)
    all_records.sort(key=lambda r: r["composite_score"], reverse=True)
    for i, r in enumerate(all_records, start=1):
        r["rank_in_run"] = i

    csv_path = METRICS_DIR / f"{run_dir.name}_ranked_candidates.csv"
    _write_csv(all_records, csv_path)
    print(f"\nRanked CSV -> {csv_path}")

    _print_run_summary(run_dir.name, all_records)
    print(f"\nTotal run time: {time.time() - t_run:.1f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute LHRR + EOP ranking metrics for one FlakeFinder run.")
    parser.add_argument("run_dir", type=Path,
                        help="extracted root of one run archive")
    parser.add_argument("--flatfield", type=Path, default=FLATFIELD_PATH,
                        metavar="PATH",
                        help="10x flatfield .npy for your microscope config "
                             "(defaults to the bundled 10x bin3 flatfield "
                             "under data/)")
    args = parser.parse_args()
    main(args.run_dir, args.flatfield)
