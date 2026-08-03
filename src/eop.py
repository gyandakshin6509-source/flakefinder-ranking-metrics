"""
eop.py: Ease of Pickup metric.

For each candidate flake, computes:
  - clearance_um         distance (µm) from candidate boundary to nearest
                         OTHER detection on the chip
  - weighted_obstruction proximity-weighted area integral of normalised contrast
                         deltas from neighbouring detections within
                         MAX_OBSTRUCTION_RADIUS_UM of the candidate centroid
  - eop_score            clearance_um / (1 + weighted_obstruction)
                         high = far from obstructions (or thin neighbours) → easy pickup
                         low  = close to thick neighbours → risky pickup

# EOP occupancy policy: ALL detections from all seg frames are included in the
# chip occupancy mask, regardless of `classification`. Non-target material
# physically obstructs stamp pickup just as much as target material.

# Resolution invariance: weighted_obstruction is computed as
#     Σ_pixels (obs[j] / d_j²) * stage_px²
# rather than a raw pixel sum. Multiplying by stage_px² turns the discrete sum
# into a proper area integral with units of (normalised-contrast / µm²) * µm² =
# dimensionless. Without this factor the sum scales as 1/stage_px², so finer
# occupancy resolution (smaller stage_px) inflates the obstruction value purely
# from increased pixel count. With the factor, the score is comparable across
# OCCUPANCY_STAGE_PX_UM settings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from io_utils import (
    apply_flatfield,
    get_frame_dims,
    get_pixel_um,
    load_scan_meta,
    lookup_detection,
    project_contour_to_stage,
)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

OCCUPANCY_STAGE_PX_UM: float = 5.0
MAX_OBSTRUCTION_RADIUS_UM: float = 500.0
BASELINE_SAMPLE_FRAMES: int = 20
EOP_MIN_CLEARANCE_UM: float = 50.0
MIN_BASELINE_PIXELS: int = 1000

_FRAME_FILE_RE = re.compile(r"frame_(\d+)\.json$")


# ---------------------------------------------------------------------------
# Substrate baseline
# ---------------------------------------------------------------------------

def estimate_substrate_baseline(
    chip_dir: Path,
    flatfield: np.ndarray,
    meta: dict[str, Any],
) -> tuple[float, bool]:
    """
    Estimate the chip-wide substrate green-channel baseline.

    Samples up to BASELINE_SAMPLE_FRAMES frames from the first and last 10% of
    the seg-frame index range (chip edges are typically sparser). For each
    sampled frame the detection contours are masked out and the mean of the
    remaining (free) green pixels is taken. The chip baseline is the median
    of those per-frame means.

    Returns
    -------
    (baseline_value, used_fallback): used_fallback is True when fewer than
    3 frames produced a usable estimate and the 5th-percentile fallback was used.
    """
    seg_files = sorted((chip_dir / "seg").glob("frame_*.json"))
    if not seg_files:
        raise FileNotFoundError(f"No seg JSONs found in {chip_dir / 'seg'}")

    n = len(seg_files)
    edge_n = max(1, n // 10)
    head = seg_files[:edge_n]
    tail = seg_files[-edge_n:]
    edge_files = head + tail

    half = max(1, BASELINE_SAMPLE_FRAMES // 2)
    head_idx = np.linspace(0, len(head) - 1, min(half, len(head))).astype(int)
    tail_idx = np.linspace(0, len(tail) - 1, min(half, len(tail))).astype(int)
    sample = [head[i] for i in head_idx] + [tail[i] for i in tail_idx]

    W, H = get_frame_dims(meta)
    per_frame_means: list[float] = []
    fallback_means: list[float] = []  # full-frame means for fallback path

    for seg_file in sample:
        m = _FRAME_FILE_RE.search(seg_file.name)
        if not m:
            continue
        frame_n = int(m.group(1))
        jpg = chip_dir / "scan_10x" / f"frame_{frame_n:04d}.jpg"
        if not jpg.exists():
            continue

        raw = cv2.imread(str(jpg))
        if raw is None:
            continue
        corrected = apply_flatfield(raw, flatfield)
        green = corrected[:, :, 1]
        fallback_means.append(float(green.mean()))

        seg_data = json.loads(seg_file.read_text())
        combined = np.zeros((H, W), dtype=np.uint8)
        for det in seg_data.get("detections", []):
            contour = np.array(det.get("contour", []), dtype=np.int32)
            if len(contour) >= 3:
                cv2.fillPoly(combined, [contour], 1)

        free_pixels = green[combined == 0]
        if free_pixels.size < MIN_BASELINE_PIXELS:
            continue

        per_frame_means.append(float(free_pixels.mean()))

    if len(per_frame_means) >= 3:
        return float(np.median(per_frame_means)), False

    if fallback_means:
        print(f"  WARNING: substrate baseline fallback "
              f"(only {len(per_frame_means)} clean frames; using 5th-pct of frame means)")
        return float(np.percentile(fallback_means, 5)), True

    raise RuntimeError("Could not compute substrate baseline: no readable frames")


# ---------------------------------------------------------------------------
# Chip occupancy + obstruction maps
# ---------------------------------------------------------------------------

def build_chip_occupancy(
    chip_dir: Path,
    flatfield: np.ndarray,
    progress_every: int = 50,
) -> dict[str, Any]:
    """
    Build the chip-wide occupancy and obstruction maps.

    Iterates every seg JSON for the chip. For each detection, fills its
    projected stage-coordinate contour into:
      - occ_map (uint8, 0/255): geometric occupancy
      - obstruction_map (float32): max-accumulated raw |Δgreen|

    Obstruction is then normalised to [0, 1] by chip-wide max.

    Parameters
    ----------
    chip_dir : Path
        Chip root (contains scan_10x/ and seg/).
    flatfield : np.ndarray
        Shape (H, W, 3) float32 BGR flatfield correction.
    progress_every : int
        Print progress every N processed seg frames.

    Returns
    -------
    dict with keys:
        occ_map           (rows, cols) uint8
        obstruction_map   (rows, cols) float32 in [0, 1]
        map_origin        (x_min_um, y_min_um)
        stage_px          µm/px scalar
        meta              parsed scan_meta.json
        substrate_baseline float: chip green-channel baseline
        baseline_fallback bool
        n_detections      total detections written
    """
    meta_path = chip_dir / "scan_10x" / "scan_meta.json"
    meta = load_scan_meta(str(meta_path))
    pixel_um = get_pixel_um(meta)
    W, H = get_frame_dims(meta)

    x_min, x_max = float(meta["x_min_um"]), float(meta["x_max_um"])
    y_min, y_max = float(meta["y_min_um"]), float(meta["y_max_um"])
    stage_px = OCCUPANCY_STAGE_PX_UM

    cols = int(np.ceil((x_max - x_min) / stage_px)) + 1
    rows = int(np.ceil((y_max - y_min) / stage_px)) + 1

    occ_map = np.zeros((rows, cols), dtype=np.uint8)
    obstruction_map = np.zeros((rows, cols), dtype=np.float32)

    print(f"  occupancy map: {rows} x {cols} px @ {stage_px:.1f} µm/px "
          f"({occ_map.nbytes / 1e6:.1f} MB uint8 + "
          f"{obstruction_map.nbytes / 1e6:.1f} MB float32)")

    print(f"  estimating substrate baseline...")
    baseline, fallback = estimate_substrate_baseline(chip_dir, flatfield, meta)
    print(f"  substrate baseline (mean green) = {baseline:.2f}"
          + (" [fallback]" if fallback else ""))

    frames_by_n = {f["n"]: f for f in meta["frames"]}

    seg_files = sorted((chip_dir / "seg").glob("frame_*.json"))
    n_total_dets = 0
    skipped_frames = 0

    map_origin = np.array([x_min, y_min], dtype=np.float32)
    map_max = np.array([cols - 1, rows - 1], dtype=np.int32)

    for i, seg_file in enumerate(seg_files):
        m = _FRAME_FILE_RE.search(seg_file.name)
        if not m:
            continue
        frame_n = int(m.group(1))
        if frame_n not in frames_by_n:
            skipped_frames += 1
            continue
        fr = frames_by_n[frame_n]

        jpg = chip_dir / "scan_10x" / f"frame_{frame_n:04d}.jpg"
        if not jpg.exists():
            skipped_frames += 1
            continue
        raw = cv2.imread(str(jpg))
        if raw is None:
            skipped_frames += 1
            continue
        corrected = apply_flatfield(raw, flatfield)
        green = corrected[:, :, 1]

        seg_data = json.loads(seg_file.read_text())
        for det in seg_data.get("detections", []):
            contour = det.get("contour")
            if contour is None or len(contour) < 3:
                continue

            local_mask = np.zeros((H, W), dtype=np.uint8)
            cnt_int32 = np.array(contour, dtype=np.int32)
            cv2.fillPoly(local_mask, [cnt_int32], 1)
            n_inside = int(local_mask.sum())
            if n_inside == 0:
                continue
            mean_green_flake = float(green[local_mask.astype(bool)].mean())
            contrast_delta = abs(mean_green_flake - baseline)

            stage_contour = project_contour_to_stage(
                contour, fr, pixel_um, W, H
            )
            map_contour = ((stage_contour - map_origin) / stage_px).astype(np.int32)
            map_contour[:, 0] = np.clip(map_contour[:, 0], 0, map_max[0])
            map_contour[:, 1] = np.clip(map_contour[:, 1], 0, map_max[1])

            cv2.fillPoly(occ_map, [map_contour], 255)

            mc_xmin, mc_ymin = int(map_contour[:, 0].min()), int(map_contour[:, 1].min())
            mc_xmax, mc_ymax = int(map_contour[:, 0].max()), int(map_contour[:, 1].max())
            crop_w = mc_xmax - mc_xmin + 1
            crop_h = mc_ymax - mc_ymin + 1
            if crop_w <= 0 or crop_h <= 0:
                continue
            sub_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
            shifted = map_contour - np.array([mc_xmin, mc_ymin], dtype=np.int32)
            cv2.fillPoly(sub_mask, [shifted], 1)
            sub_mask_b = sub_mask.astype(bool)

            view = obstruction_map[mc_ymin:mc_ymax + 1, mc_xmin:mc_xmax + 1]
            np.maximum(view, np.where(sub_mask_b, contrast_delta, 0.0).astype(np.float32),
                       out=view)

            n_total_dets += 1

        if (i + 1) % progress_every == 0:
            print(f"  processed {i + 1}/{len(seg_files)} seg frames, "
                  f"{n_total_dets} detections so far")

    max_contrast = float(obstruction_map.max())
    if max_contrast > 0:
        obstruction_map /= max_contrast

    print(f"  occupancy build complete: {n_total_dets} detections, "
          f"{skipped_frames} frames skipped (no jpg/meta entry), "
          f"max raw contrast={max_contrast:.2f}")

    return {
        "occ_map": occ_map,
        "obstruction_map": obstruction_map,
        "map_origin": map_origin,
        "stage_px": stage_px,
        "meta": meta,
        "substrate_baseline": baseline,
        "baseline_fallback": fallback,
        "n_detections": n_total_dets,
        "max_raw_contrast": max_contrast,
    }


# ---------------------------------------------------------------------------
# Per-flake EOP
# ---------------------------------------------------------------------------

def compute_eop(
    chip_dir: Path,
    label: str,
    chip_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute EOP for one candidate flake.

    Parameters
    ----------
    chip_dir : Path
    label : str
        Revisit candidate label, e.g. 'rank01_frame_0432_d1'.
    chip_data : dict
        Output of build_chip_occupancy(...).

    Returns
    -------
    dict with keys:
        clearance_um, weighted_obstruction, eop_score,
        clearance_warning, cand_pixels, stage_center_um,
        eop_skip_reason (None on success).
    """
    occ_map = chip_data["occ_map"]
    obstruction_map = chip_data["obstruction_map"]
    map_origin = chip_data["map_origin"]
    stage_px = chip_data["stage_px"]
    meta = chip_data["meta"]
    pixel_um = get_pixel_um(meta)
    W, H = get_frame_dims(meta)
    rows, cols = occ_map.shape

    det, frame_entry, _ = lookup_detection(chip_dir, label)
    contour = det["contour"]
    cx_px, cy_px = float(det["center"][0]), float(det["center"][1])
    stage_cx = frame_entry["x_um"] + (cx_px - W / 2.0) * pixel_um
    stage_cy = frame_entry["y_um"] + (cy_px - H / 2.0) * pixel_um

    stage_contour = project_contour_to_stage(contour, frame_entry, pixel_um, W, H)
    cand_map = ((stage_contour - map_origin) / stage_px).astype(np.int32)
    cand_map[:, 0] = np.clip(cand_map[:, 0], 0, cols - 1)
    cand_map[:, 1] = np.clip(cand_map[:, 1], 0, rows - 1)

    cand_mask_u8 = np.zeros((rows, cols), dtype=np.uint8)
    cv2.fillPoly(cand_mask_u8, [cand_map], 1)
    cand_mask = cand_mask_u8.astype(bool)
    n_cand = int(cand_mask.sum())

    if n_cand == 0:
        return {
            "clearance_um": 0.0,
            "weighted_obstruction": 0.0,
            "eop_score": 0.0,
            "clearance_warning": True,
            "cand_pixels": 0,
            "stage_center_um": (round(stage_cx, 3), round(stage_cy, 3)),
            "eop_skip_reason": "candidate_outside_map",
        }

    # Clearance via distance transform on free space (with candidate removed)
    occ_without = occ_map.copy()
    occ_without[cand_mask] = 0
    # ~occ_without: free (0)→255 (foreground for distanceTransform), occ (255)→0 (barrier)
    dist_map = cv2.distanceTransform(~occ_without, cv2.DIST_L2, 5)

    eroded = cv2.erode(cand_mask_u8, np.ones((3, 3), np.uint8))
    boundary = cand_mask & ~eroded.astype(bool)
    if not boundary.any():
        boundary = cand_mask

    clearance_px = float(dist_map[boundary].min())
    clearance_um = clearance_px * stage_px

    # Weighted obstruction within radius around candidate centroid
    radius_px = int(np.ceil(MAX_OBSTRUCTION_RADIUS_UM / stage_px))
    cx_map = (stage_cx - map_origin[0]) / stage_px
    cy_map = (stage_cy - map_origin[1]) / stage_px
    cx_idx = int(round(cx_map))
    cy_idx = int(round(cy_map))

    x0 = max(0, cx_idx - radius_px)
    x1 = min(cols, cx_idx + radius_px + 1)
    y0 = max(0, cy_idx - radius_px)
    y1 = min(rows, cy_idx + radius_px + 1)

    obs_crop = obstruction_map[y0:y1, x0:x1]
    cand_crop = cand_mask[y0:y1, x0:x1]

    yy, xx = np.indices(obs_crop.shape)
    dx_um = (xx + x0 - cx_map) * stage_px
    dy_um = (yy + y0 - cy_map) * stage_px
    dist_um = np.sqrt(dx_um ** 2 + dy_um ** 2)

    valid = (
        (obs_crop > 0)
        & (~cand_crop)
        & (dist_um <= MAX_OBSTRUCTION_RADIUS_UM)
    )
    safe_dist = np.maximum(dist_um, 1.0)
    # Multiply by pixel area (stage_px²) to convert pixel-sum into an area
    # integral; otherwise the sum scales with 1/stage_px² and the score
    # becomes resolution-dependent. See module docstring.
    pixel_area_um2 = stage_px ** 2
    weighted_obstruction = float(
        (obs_crop[valid] / safe_dist[valid] ** 2).sum() * pixel_area_um2
    )

    eop_score = clearance_um / (1.0 + weighted_obstruction)
    clearance_warning = clearance_um < EOP_MIN_CLEARANCE_UM

    return {
        "clearance_um": round(clearance_um, 3),
        "weighted_obstruction": round(weighted_obstruction, 6),
        "eop_score": round(eop_score, 3),
        "clearance_warning": bool(clearance_warning),
        "cand_pixels": n_cand,
        "stage_center_um": (round(stage_cx, 3), round(stage_cy, 3)),
        "eop_skip_reason": None,
    }


# ---------------------------------------------------------------------------
# Diagnostic figures
# ---------------------------------------------------------------------------

def save_chip_occupancy_qc(
    chip_data: dict[str, Any],
    out_path: Path,
    title: str = "",
    downsample: int = 4,
) -> None:
    """
    Save a QC PNG of the full chip occupancy + obstruction maps (downsampled).

    Two side-by-side panels: occupancy (binary) and obstruction (heatmap).
    """
    occ = chip_data["occ_map"][::downsample, ::downsample]
    obs = chip_data["obstruction_map"][::downsample, ::downsample]
    map_origin = chip_data["map_origin"]
    stage_px = chip_data["stage_px"]

    rows, cols = occ.shape
    extent_um = (
        float(map_origin[0]),
        float(map_origin[0] + cols * downsample * stage_px),
        float(map_origin[1] + rows * downsample * stage_px),
        float(map_origin[1]),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(title or "Chip occupancy QC", fontsize=12)

    ax = axes[0]
    ax.imshow(occ, cmap="gray", extent=extent_um, aspect="equal", interpolation="nearest")
    ax.set_title(f"Occupancy (white=detection, n={chip_data['n_detections']})")
    ax.set_xlabel("stage x (µm)")
    ax.set_ylabel("stage y (µm)")

    ax = axes[1]
    ax.imshow(obs, cmap="hot", extent=extent_um, aspect="equal", interpolation="nearest",
              vmin=0.0, vmax=1.0)
    ax.set_title("Obstruction (normalised |Δgreen|, max-accumulated)")
    ax.set_xlabel("stage x (µm)")
    ax.set_ylabel("stage y (µm)")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_eop_figure(
    chip_dir: Path,
    label: str,
    chip_data: dict[str, Any],
    eop_result: dict[str, Any],
    out_path: Path,
    crop_radius_um: float = 600.0,
) -> None:
    """
    Save a per-flake EOP diagnostic PNG.

    Shows a crop of the occupancy map around the candidate, with the candidate
    drawn in green and all other detections in red. Includes a metric annotation.
    """
    occ_map = chip_data["occ_map"]
    obstruction_map = chip_data["obstruction_map"]
    map_origin = chip_data["map_origin"]
    stage_px = chip_data["stage_px"]
    meta = chip_data["meta"]
    pixel_um = get_pixel_um(meta)
    W, H = get_frame_dims(meta)
    rows, cols = occ_map.shape

    det, frame_entry, _ = lookup_detection(chip_dir, label)
    cx_px, cy_px = float(det["center"][0]), float(det["center"][1])
    stage_cx = frame_entry["x_um"] + (cx_px - W / 2.0) * pixel_um
    stage_cy = frame_entry["y_um"] + (cy_px - H / 2.0) * pixel_um
    stage_contour = project_contour_to_stage(det["contour"], frame_entry, pixel_um, W, H)
    cand_map = ((stage_contour - map_origin) / stage_px).astype(np.int32)
    cand_map[:, 0] = np.clip(cand_map[:, 0], 0, cols - 1)
    cand_map[:, 1] = np.clip(cand_map[:, 1], 0, rows - 1)

    cand_mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.fillPoly(cand_mask, [cand_map], 1)
    cand_mask_b = cand_mask.astype(bool)

    radius_px = int(np.ceil(crop_radius_um / stage_px))
    cx_map = (stage_cx - map_origin[0]) / stage_px
    cy_map = (stage_cy - map_origin[1]) / stage_px
    cx_idx = int(round(cx_map))
    cy_idx = int(round(cy_map))
    x0 = max(0, cx_idx - radius_px)
    x1 = min(cols, cx_idx + radius_px + 1)
    y0 = max(0, cy_idx - radius_px)
    y1 = min(rows, cy_idx + radius_px + 1)

    occ_crop = occ_map[y0:y1, x0:x1]
    cand_crop = cand_mask_b[y0:y1, x0:x1]
    obs_crop = obstruction_map[y0:y1, x0:x1]

    rgb = np.zeros((*occ_crop.shape, 3), dtype=np.uint8)
    rgb[occ_crop > 0] = [200, 30, 30]
    rgb[cand_crop] = [40, 220, 40]

    extent_um = (
        float(map_origin[0] + x0 * stage_px),
        float(map_origin[0] + x1 * stage_px),
        float(map_origin[1] + y1 * stage_px),
        float(map_origin[1] + y0 * stage_px),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(label, fontsize=11)

    ax = axes[0]
    ax.imshow(rgb, extent=extent_um, interpolation="nearest", aspect="equal")
    ax.set_title("Occupancy (green=candidate, red=other detections)")
    ax.set_xlabel("stage x (µm)")
    ax.set_ylabel("stage y (µm)")
    circ = mpatches.Circle(
        (stage_cx, stage_cy), MAX_OBSTRUCTION_RADIUS_UM,
        edgecolor="cyan", facecolor="none", linewidth=1.0, linestyle="--",
    )
    ax.add_patch(circ)

    ax = axes[1]
    ax.imshow(obs_crop, cmap="hot", extent=extent_um, interpolation="nearest",
              aspect="equal", vmin=0.0, vmax=1.0)
    ax.set_title("Obstruction map (normalised contrast |Δgreen|)")
    ax.set_xlabel("stage x (µm)")
    ax.set_ylabel("stage y (µm)")
    circ2 = mpatches.Circle(
        (stage_cx, stage_cy), MAX_OBSTRUCTION_RADIUS_UM,
        edgecolor="cyan", facecolor="none", linewidth=1.0, linestyle="--",
    )
    ax.add_patch(circ2)

    note = (
        f"clearance={eop_result['clearance_um']:.0f} µm   "
        f"obstruction={eop_result['weighted_obstruction']:.3f}   "
        f"eop={eop_result['eop_score']:.1f}   "
        f"warn={eop_result['clearance_warning']}"
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=11)

    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
