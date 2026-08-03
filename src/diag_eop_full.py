"""
diag_eop_full.py — Three EOP validation diagnostics for chip_0 hBN.

Order:
  1. Build chip occupancy at 5 µm/px (default).
  2. Run Diag 2 (neighbour table for rank02) and Diag 3 (occupancy zoom)
     using the 5 µm map.
  3. Capture the px=5.0 row of Diag 1.
  4. Free the 5 µm map; rebuild at 2 µm and 1 µm to fill the rest of Diag 1.

Usage:
    python src/diag_eop_full.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import eop
from eop import build_chip_occupancy, compute_eop
from io_utils import (
    apply_flatfield,
    get_frame_dims,
    get_pixel_um,
    lookup_detection,
    parse_revisit_label,
    project_contour_to_stage,
)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
CHIP_DIR = DATA_ROOT / "_extracted" / "run_20260505_1616" / "chip_0"
FLATFIELD_NPY = DATA_ROOT / "flakes data" / "flakes data" / "flatfields" / "flatfield_10x_bin3.npy"
OUT_DIR = PROJECT_ROOT / "outputs" / "eop"

VAL_FLAKES = [
    ("small",  "rank04_frame_0389_d28"),
    ("median", "rank13_frame_0149_d5"),
    ("large",  "rank02_frame_0257_d2"),
]
RANK02_LABEL = "rank02_frame_0257_d2"
NEIGHBOR_RADIUS_UM = 50.0


# ---------------------------------------------------------------------------
# Diagnostic 2 — detections within 50 µm of rank02 boundary
# ---------------------------------------------------------------------------

def run_diag2() -> None:
    print("\n" + "=" * 78)
    print(f"DIAG 2 — detections within {NEIGHBOR_RADIUS_UM:.0f} µm of {RANK02_LABEL} boundary")
    print("=" * 78)

    cand_det, cand_frame, meta = lookup_detection(CHIP_DIR, RANK02_LABEL)
    pixel_um = get_pixel_um(meta)
    W, H = get_frame_dims(meta)
    _, cand_frame_n, cand_d_id = parse_revisit_label(RANK02_LABEL)

    cx_px, cy_px = float(cand_det["center"][0]), float(cand_det["center"][1])
    cand_cx = cand_frame["x_um"] + (cx_px - W / 2.0) * pixel_um
    cand_cy = cand_frame["y_um"] + (cy_px - H / 2.0) * pixel_um
    cand_poly = project_contour_to_stage(cand_det["contour"], cand_frame, pixel_um, W, H)

    # frame_0257 stage extent (for adjacency / overlap check)
    f0_xmin = cand_frame["x_um"] - W / 2.0 * pixel_um
    f0_xmax = cand_frame["x_um"] + W / 2.0 * pixel_um
    f0_ymin = cand_frame["y_um"] - H / 2.0 * pixel_um
    f0_ymax = cand_frame["y_um"] + H / 2.0 * pixel_um

    # 1 µm/px raster of cand mask, distance to candidate boundary
    pad_um = NEIGHBOR_RADIUS_UM + 50.0
    xmin = float(cand_poly[:, 0].min() - pad_um)
    xmax = float(cand_poly[:, 0].max() + pad_um)
    ymin = float(cand_poly[:, 1].min() - pad_um)
    ymax = float(cand_poly[:, 1].max() + pad_um)
    raster_px_um = 1.0
    n_cols = int(np.ceil((xmax - xmin) / raster_px_um)) + 1
    n_rows = int(np.ceil((ymax - ymin) / raster_px_um)) + 1
    cand_canvas = np.zeros((n_rows, n_cols), dtype=np.uint8)
    cp = ((cand_poly - np.array([xmin, ymin], dtype=np.float32)) / raster_px_um).astype(np.int32)
    cv2.fillPoly(cand_canvas, [cp], 1)
    eroded = cv2.erode(cand_canvas, np.ones((3, 3), np.uint8))
    boundary = (cand_canvas > 0) & (eroded == 0)
    bg_for_dt = np.where(boundary, 0, 255).astype(np.uint8)
    dist_to_boundary = cv2.distanceTransform(bg_for_dt, cv2.DIST_L2, 5)  # px = µm

    print(f"  cand source frame: n={cand_frame_n}, centroid=({cand_cx:.1f}, {cand_cy:.1f}) µm")
    print(f"  frame_0257 extent: x=[{f0_xmin:.1f}, {f0_xmax:.1f}], "
          f"y=[{f0_ymin:.1f}, {f0_ymax:.1f}] µm")

    seg_files = sorted((CHIP_DIR / "seg").glob("frame_*.json"))
    frames_by_n = {f["n"]: f for f in meta["frames"]}

    rows_out: list[dict] = []
    for sf in seg_files:
        m = eop._FRAME_FILE_RE.search(sf.name)
        if not m:
            continue
        fn = int(m.group(1))
        if fn not in frames_by_n:
            continue
        fr = frames_by_n[fn]
        seg = json.loads(sf.read_text())
        for d_idx, det in enumerate(seg.get("detections", [])):
            if fn == cand_frame_n and d_idx == cand_d_id:
                continue
            cnt = det.get("contour")
            if cnt is None or len(cnt) < 3:
                continue
            poly_stage = project_contour_to_stage(cnt, fr, pixel_um, W, H)
            if (poly_stage[:, 0].min() > xmax or poly_stage[:, 0].max() < xmin or
                poly_stage[:, 1].min() > ymax or poly_stage[:, 1].max() < ymin):
                continue
            cu = ((poly_stage - np.array([xmin, ymin], dtype=np.float32)) / raster_px_um).astype(np.int32)
            cu[:, 0] = np.clip(cu[:, 0], 0, n_cols - 1)
            cu[:, 1] = np.clip(cu[:, 1], 0, n_rows - 1)
            d_min = float(dist_to_boundary[cu[:, 1], cu[:, 0]].min())
            if d_min > NEIGHBOR_RADIUS_UM:
                continue

            cxp, cyp = float(det["center"][0]), float(det["center"][1])
            sx = fr["x_um"] + (cxp - W / 2.0) * pixel_um
            sy = fr["y_um"] + (cyp - H / 2.0) * pixel_um
            d_centroid = float(np.hypot(sx - cand_cx, sy - cand_cy))

            f_xmin = fr["x_um"] - W / 2.0 * pixel_um
            f_xmax = fr["x_um"] + W / 2.0 * pixel_um
            f_ymin = fr["y_um"] - H / 2.0 * pixel_um
            f_ymax = fr["y_um"] + H / 2.0 * pixel_um
            frame_overlaps = (f_xmin < f0_xmax and f_xmax > f0_xmin and
                              f_ymin < f0_ymax and f_ymax > f0_ymin) and (fn != cand_frame_n)
            centroid_in_0257 = (f0_xmin <= sx <= f0_xmax and
                                f0_ymin <= sy <= f0_ymax) and (fn != cand_frame_n)

            rows_out.append({
                "frame_n": fn,
                "d_idx": d_idx,
                "stage_x": sx,
                "stage_y": sy,
                "size_um2": det.get("size_um2"),
                "thickness_nm": det.get("thickness_nm"),
                "d_boundary_um": d_min,
                "d_centroid_um": d_centroid,
                "classification": det.get("classification") or "?",
                "frame_overlaps_0257": frame_overlaps,
                "centroid_in_0257": centroid_in_0257,
            })

    rows_out.sort(key=lambda r: r["d_centroid_um"])

    print(
        f"\n  {'frame':>5} {'d':>3} {'stage_x':>9} {'stage_y':>9} "
        f"{'size_um2':>9} {'thk':>5} {'d_bnd':>6} {'d_cnt':>6} "
        f"{'class':<12} {'overlap':>7} {'in_0257':>8}"
    )
    print("  " + "-" * 102)
    for r in rows_out:
        size_str = f"{r['size_um2']:.0f}" if r["size_um2"] is not None else "?"
        thk = r["thickness_nm"]
        thk_str = f"{thk:.1f}" if isinstance(thk, (int, float)) else "?"
        print(
            f"  {r['frame_n']:>5} {r['d_idx']:>3} "
            f"{r['stage_x']:>9.1f} {r['stage_y']:>9.1f} "
            f"{size_str:>9} {thk_str:>5} "
            f"{r['d_boundary_um']:>6.1f} {r['d_centroid_um']:>6.1f} "
            f"{r['classification']:<12} "
            f"{str(r['frame_overlaps_0257']):>7} {str(r['centroid_in_0257']):>8}"
        )
    print(f"  total: {len(rows_out)} obstruction detections within "
          f"{NEIGHBOR_RADIUS_UM:.0f} µm of cand boundary")

    n_overlap = sum(1 for r in rows_out if r["frame_overlaps_0257"])
    n_in_extent = sum(1 for r in rows_out if r["centroid_in_0257"])
    print(f"  -> in spatially-overlapping frames (!= 0257): {n_overlap}")
    print(f"  -> centroids inside frame_0257 stage extent (!= 0257): {n_in_extent}")
    if n_in_extent > 0:
        print("  WARN: multi-frame coverage detected -- detections from other frames sit "
              "inside frame_0257's FOV (likely double-coverage of cand neighbourhood)")
    else:
        print("  OK: no double-coverage -- every nearby detection comes from frame_0257 itself")


# ---------------------------------------------------------------------------
# Diagnostic 3 — occupancy zoom vs corrected frame side-by-side
# ---------------------------------------------------------------------------

def run_diag3(chip_data: dict, flatfield: np.ndarray) -> None:
    print("\n" + "=" * 78)
    print(f"DIAG 3 — corrected frame_0257 vs occupancy at same FOV — {RANK02_LABEL}")
    print("=" * 78)

    cand_det, cand_frame, meta = lookup_detection(CHIP_DIR, RANK02_LABEL)
    pixel_um = get_pixel_um(meta)
    W, H = get_frame_dims(meta)

    sx0 = cand_frame["x_um"] - W / 2.0 * pixel_um
    sx1 = cand_frame["x_um"] + W / 2.0 * pixel_um
    sy0 = cand_frame["y_um"] - H / 2.0 * pixel_um
    sy1 = cand_frame["y_um"] + H / 2.0 * pixel_um
    print(f"  frame_0257 extent: x=[{sx0:.1f}, {sx1:.1f}] µm  y=[{sy0:.1f}, {sy1:.1f}] µm "
          f"(W={sx1 - sx0:.0f} µm, H={sy1 - sy0:.0f} µm)")

    jpg = CHIP_DIR / "scan_10x" / f"frame_{cand_frame['n']:04d}.jpg"
    raw = cv2.imread(str(jpg))
    if raw is None:
        raise FileNotFoundError(f"Missing jpg: {jpg}")
    corrected = apply_flatfield(raw, flatfield)
    rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

    cand_stage = project_contour_to_stage(cand_det["contour"], cand_frame, pixel_um, W, H)

    occ_map = chip_data["occ_map"]
    map_origin = chip_data["map_origin"]
    stage_px = chip_data["stage_px"]
    rows_n, cols_n = occ_map.shape
    mx0 = max(0, int(np.floor((sx0 - map_origin[0]) / stage_px)))
    mx1 = min(cols_n, int(np.ceil((sx1 - map_origin[0]) / stage_px)) + 1)
    my0 = max(0, int(np.floor((sy0 - map_origin[1]) / stage_px)))
    my1 = min(rows_n, int(np.ceil((sy1 - map_origin[1]) / stage_px)) + 1)
    occ_crop = occ_map[my0:my1, mx0:mx1]
    occ_extent = (
        float(map_origin[0] + mx0 * stage_px),
        float(map_origin[0] + mx1 * stage_px),
        float(map_origin[1] + my1 * stage_px),
        float(map_origin[1] + my0 * stage_px),
    )

    out_path = OUT_DIR / "rank02_occupancy_zoom.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"{RANK02_LABEL} — frame_0257 corrected vs chip occupancy "
        f"(FOV = {sx1 - sx0:.0f} × {sy1 - sy0:.0f} µm)",
        fontsize=11,
    )

    ax = axes[0]
    ax.imshow(rgb, extent=(sx0, sx1, sy1, sy0), aspect="equal", interpolation="nearest")
    ax.plot(cand_stage[:, 0].tolist() + [float(cand_stage[0, 0])],
            cand_stage[:, 1].tolist() + [float(cand_stage[0, 1])],
            color="lime", linewidth=1.5)
    ax.set_title("Corrected frame_0257 + flake outline")
    ax.set_xlabel("stage x (µm)")
    ax.set_ylabel("stage y (µm)")
    ax.set_xlim(sx0, sx1)
    ax.set_ylim(sy1, sy0)

    ax = axes[1]
    ax.imshow(occ_crop, cmap="gray", extent=occ_extent,
              aspect="equal", interpolation="nearest", vmin=0, vmax=255)
    ax.plot(cand_stage[:, 0].tolist() + [float(cand_stage[0, 0])],
            cand_stage[:, 1].tolist() + [float(cand_stage[0, 1])],
            color="lime", linewidth=1.5)
    ax.set_title(f"Chip occupancy (5 µm/px) — same FOV")
    ax.set_xlabel("stage x (µm)")
    ax.set_ylabel("stage y (µm)")
    ax.set_xlim(sx0, sx1)
    ax.set_ylim(sy1, sy0)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Diagnostic 1 — clearance vs occupancy resolution (5, 2, 1 µm/px)
# ---------------------------------------------------------------------------

def _eop_row(label: str, chip_data: dict) -> tuple[float, float]:
    r = compute_eop(CHIP_DIR, label, chip_data)
    return float(r["clearance_um"]), float(r["eop_score"])


def _print_diag1_table(rows: dict) -> None:
    print("\n" + "=" * 78)
    print("DIAG 1 — Resolution sensitivity (clearance µm | EOP score) per OCCUPANCY_STAGE_PX_UM")
    print("=" * 78)
    head = f"  {'flake':<32} {'px=5.0':>20} {'px=2.0':>20} {'px=1.0':>20}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for _, label in VAL_FLAKES:
        c5, e5 = rows[label][5.0]
        c2, e2 = rows[label][2.0]
        c1, e1 = rows[label][1.0]
        cell5 = f"{c5:>5.1f} µm | {e5:>5.1f}"
        cell2 = f"{c2:>5.1f} µm | {e2:>5.1f}"
        cell1 = f"{c1:>5.1f} µm | {e1:>5.1f}"
        print(f"  {label:<32} {cell5:>20} {cell2:>20} {cell1:>20}")
    print("\n  Interpretation:")
    print("  - clearance flat at the floor across 5→2→1 → genuine adjacency (touching)")
    print("  - clearance grows substantially → 5 µm/px was a resolution artifact")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CHIP_DIR.exists():
        print(f"ERROR: chip_dir not found: {CHIP_DIR}")
        sys.exit(1)
    flatfield = np.load(str(FLATFIELD_NPY)).astype(np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    diag1_rows: dict[str, dict[float, tuple[float, float]]] = {label: {} for _, label in VAL_FLAKES}

    print("=" * 78)
    print("Building chip_0 occupancy at OCCUPANCY_STAGE_PX_UM = 5.0 (default)")
    print("=" * 78)
    eop.OCCUPANCY_STAGE_PX_UM = 5.0
    t0 = time.time()
    chip_data = build_chip_occupancy(CHIP_DIR, flatfield, progress_every=100)
    print(f"  build time: {time.time() - t0:.1f} s")

    # Diag 1 (px=5 row)
    for _, label in VAL_FLAKES:
        c, e = _eop_row(label, chip_data)
        diag1_rows[label][5.0] = (c, e)

    # Diag 2
    run_diag2()

    # Diag 3 (uses 5 µm chip_data)
    run_diag3(chip_data, flatfield)

    # Free 5 µm
    del chip_data
    gc.collect()

    # Diag 1 — px=2 and px=1
    for px in (2.0, 1.0):
        print("\n" + "=" * 78)
        print(f"Building chip_0 occupancy at OCCUPANCY_STAGE_PX_UM = {px}")
        print("=" * 78)
        eop.OCCUPANCY_STAGE_PX_UM = px
        t0 = time.time()
        cd = build_chip_occupancy(CHIP_DIR, flatfield, progress_every=100)
        print(f"  build time: {time.time() - t0:.1f} s")
        for _, label in VAL_FLAKES:
            c, e = _eop_row(label, cd)
            diag1_rows[label][px] = (c, e)
            print(f"  {label}: clearance={c:.1f} µm, eop={e:.1f}")
        del cd
        gc.collect()

    _print_diag1_table(diag1_rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
