"""
run_all_metrics.py: driver that runs flake_metrics.py on both available
extracted runs (hBN + graphene) sequentially, then prints combined summary
statistics across all candidates.

Usage:
    python src/run_all_metrics.py
"""

from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import flake_metrics

PROJECT_ROOT = Path(__file__).parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "_extracted"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

RUNS = [
    "run_20260505_1616",  # hBN, 6 chips
    "run_20260504_1026",  # graphene, 2 chips
]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _maybe_float(v):
    if v is None or v == "" or v == "None":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _maybe_bool(v):
    if v is None or v == "" or v == "None":
        return None
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() == "true"


def _print_combined_summary(all_rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("COMBINED SUMMARY (all runs, all chips)")
    print("=" * 78)
    n = len(all_rows)
    print(f"  total candidates processed: {n}")
    if n == 0:
        return

    n_warn_true = sum(1 for r in all_rows if _maybe_bool(r.get("clearance_warning")) is True)
    n_warn_false = sum(1 for r in all_rows if _maybe_bool(r.get("clearance_warning")) is False)
    n_warn_none = sum(1 for r in all_rows if _maybe_bool(r.get("clearance_warning")) is None)
    print(f"  clearance_warning=True    : {n_warn_true}")
    print(f"  clearance_warning=False   : {n_warn_false}")
    print(f"  clearance_warning=None    : {n_warn_none}")

    flag_counts: dict[str, int] = {}
    for r in all_rows:
        f = r.get("lhrr_quality_flag") or "None"
        flag_counts[f] = flag_counts.get(f, 0) + 1
    print("  lhrr_quality_flag counts  :")
    for k in sorted(flag_counts):
        print(f"    {k:<10} {flag_counts[k]}")

    print("\n  composite score by material:")
    print(f"    {'material':<24} {'n':>4} {'mean':>8} {'median':>8} {'max':>8}")
    by_mat: dict[str, list[float]] = {}
    for r in all_rows:
        c = _maybe_float(r.get("composite_score"))
        if c is None:
            continue
        by_mat.setdefault(r.get("material", "?"), []).append(c)
    for mat, comps in sorted(by_mat.items()):
        if not comps:
            continue
        print(f"    {mat:<24} {len(comps):>4} "
              f"{statistics.mean(comps):>8.4f} "
              f"{statistics.median(comps):>8.4f} "
              f"{max(comps):>8.4f}")


def main() -> None:
    t0 = time.time()
    all_rows: list[dict] = []
    for run_name in RUNS:
        run_dir = EXTRACTED_DIR / run_name
        if not run_dir.exists():
            print(f"SKIP: {run_dir} not found")
            continue
        print("\n" + "#" * 78)
        print(f"# RUN: {run_name}")
        print("#" * 78)
        flake_metrics.main(run_dir)
        csv_path = METRICS_DIR / f"{run_name}_ranked_candidates.csv"
        all_rows.extend(_read_csv(csv_path))

    _print_combined_summary(all_rows)
    print(f"\nTotal wall-clock: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
