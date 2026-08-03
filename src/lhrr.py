"""
lhrr.py: Largest Homogeneous Rectangular Region metric.

For each candidate flake, finds the largest axis-aligned rectangle fully
contained within the clean (low-variance) region of the flake interior.
Reports lhrr_area_px, lhrr_area_um2, lhrr_fraction, and bbox in both
frame-pixel and stage-µm coordinates.

Classification policy:
  LHRR is the "is this flake worth stacking?" signal. If a detection's
  classification is not in LHRR_CLASSIFICATION_ALLOWLIST[material],
  all output fields are set to None and lhrr_skip_reason is set to
  "classification_mismatch". Setting the allowlist value to None accepts
  all classifications (default: the lab should tighten once the
  classifier's full output vocabulary is known).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from io_utils import (
    apply_flatfield,
    build_flake_mask,
    get_frame_dims,
    get_pixel_um,
    lookup_detection,
    pixel_bbox_to_stage,
)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

VARIANCE_KERNEL_UM: float = 5.0
VARIANCE_PERCENTILE: int = 50
CROP_PAD_PX: int = 30

# Minimum effective LHRR side length (µm) for the "usable" quality flag,
# per material preset. The default value (20 µm) was calibrated against hBN;
# graphene flakes are typically smaller, so a tighter threshold differentiates
# the population. Both values are conservative, and the lab should re-tune to
# their actual stamp tolerances.
LHRR_USEFUL_THRESHOLD_UM_BY_MATERIAL: dict[str, float] = {
    "hbn_medium":         20.0,
    "graphene_thin_90nm": 10.0,
}
DEFAULT_LHRR_USEFUL_THRESHOLD_UM: float = 15.0
_warned_unknown_materials: set[str] = set()

LHRR_CLASSIFICATION_ALLOWLIST: dict[str, list[str] | None] = {
    "hbn_medium":         None,
    "graphene_thin_90nm": None,
}


def _useful_threshold_um(material: str | None) -> float:
    """
    Look up the LHRR usable-side threshold for a given material preset.
    Falls back to DEFAULT_LHRR_USEFUL_THRESHOLD_UM and prints a one-time
    warning if the material is unknown.
    """
    if material is not None and material in LHRR_USEFUL_THRESHOLD_UM_BY_MATERIAL:
        return LHRR_USEFUL_THRESHOLD_UM_BY_MATERIAL[material]
    key = material if material is not None else "<None>"
    if key not in _warned_unknown_materials:
        _warned_unknown_materials.add(key)
        print(f"  WARNING: no LHRR_USEFUL_THRESHOLD_UM_BY_MATERIAL entry "
              f"for material={key!r}; falling back to "
              f"DEFAULT_LHRR_USEFUL_THRESHOLD_UM={DEFAULT_LHRR_USEFUL_THRESHOLD_UM} µm")
    return DEFAULT_LHRR_USEFUL_THRESHOLD_UM


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_lhrr(
    chip_dir: Path,
    label: str,
    flatfield: np.ndarray,
    material: str | None = None,
    variance_percentile: int | None = None,
) -> dict[str, Any]:
    """
    Compute LHRR for one candidate flake.

    Parameters
    ----------
    chip_dir : Path
        Chip root directory (contains scan_10x/ and seg/).
    label : str
        Revisit candidate label, e.g. 'rank01_frame_0432_d1'.
    flatfield : np.ndarray
        Flatfield correction array, shape (H, W, 3), float32, BGR.
    material : str | None
        Material preset string (from summary.json). If None, no classification
        filtering is applied.
    variance_percentile : int | None
        Override VARIANCE_PERCENTILE for this call. None uses the module default.

    Returns
    -------
    dict with keys:
        lhrr_area_px, lhrr_area_um2, lhrr_fraction,
        lhrr_bbox_frame, lhrr_bbox_stage,
        variance_threshold_used, kernel_size_px,
        lhrr_skip_reason  (None on success, string on skip)
    """
    result, _ = _compute_lhrr_impl(chip_dir, label, flatfield, material,
                                   diagnostics=False,
                                   variance_percentile=variance_percentile)
    return result


def save_lhrr_figure(
    chip_dir: Path,
    label: str,
    flatfield: np.ndarray,
    out_path: Path,
    material: str | None = None,
    variance_percentile: int | None = None,
) -> dict[str, Any]:
    """
    Compute LHRR and save a 4-panel diagnostic PNG to out_path.

    Returns the same result dict as compute_lhrr.

    Parameters
    ----------
    chip_dir : Path
    label : str
    flatfield : np.ndarray
    out_path : Path
        Destination .png file path.
    material : str | None
    variance_percentile : int | None
        Override VARIANCE_PERCENTILE for this call. None uses the module default.
    """
    result, diag = _compute_lhrr_impl(chip_dir, label, flatfield, material,
                                      diagnostics=True,
                                      variance_percentile=variance_percentile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(label, result, diag, out_path)
    return result


def max_rect_in_binary_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Find the largest axis-aligned rectangle of True pixels in a 2-D bool mask.

    Uses the O(rows × cols) maximal-histogram method.

    Parameters
    ----------
    mask : np.ndarray
        2-D bool array, shape (R, C).

    Returns
    -------
    (col_left, row_top, width, height) in mask coordinates, or None if the
    mask contains no True pixels.
    """
    R, C = mask.shape
    h_map = np.zeros(C, dtype=np.int32)
    best_area = 0
    best_rect: tuple[int, int, int, int] | None = None

    for r in range(R):
        h_map = np.where(mask[r], h_map + 1, 0)
        rect = _largest_rect_in_histogram(h_map, r)
        if rect is not None:
            area = rect[2] * rect[3]
            if area > best_area:
                best_area = area
                best_rect = rect

    return best_rect


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _largest_rect_in_histogram(
    heights: np.ndarray,
    row_idx: int,
) -> tuple[int, int, int, int] | None:
    """
    Largest rectangle in a histogram via stack algorithm, O(N).

    Parameters
    ----------
    heights : 1-D int array
        Height of consecutive True pixels above each column at the current row.
    row_idx : int
        Current row index in the original mask.

    Returns
    -------
    (col_left, row_top, width, height) or None if all heights are zero.
    """
    stack: list[int] = []
    best_area = 0
    best: tuple[int, int, int, int] | None = None
    N = len(heights)

    for i in range(N + 1):
        h = int(heights[i]) if i < N else 0
        while stack and heights[stack[-1]] > h:
            height = int(heights[stack.pop()])
            width = i if not stack else i - stack[-1] - 1
            col_left = 0 if not stack else stack[-1] + 1
            area = height * width
            if area > best_area:
                best_area = area
                best = (col_left, row_idx - height + 1, width, height)
        stack.append(i)

    return best


def _odd_round(value: float) -> int:
    """Round value to nearest odd integer >= 3."""
    n = max(3, int(round(value)))
    return n if n % 2 == 1 else n + 1


def _null_result(skip_reason: str) -> dict[str, Any]:
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


def _compute_lhrr_impl(
    chip_dir: Path,
    label: str,
    flatfield: np.ndarray,
    material: str | None,
    diagnostics: bool,
    variance_percentile: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """
    Core implementation. Returns (result_dict, diag_dict | None).
    diag_dict is only populated when diagnostics=True.
    """
    # Step 0: detection lookup + classification gate
    det, frame_entry, meta = lookup_detection(chip_dir, label)

    if material is not None:
        allowlist = LHRR_CLASSIFICATION_ALLOWLIST.get(material)
        if allowlist is not None:
            cls = det.get("classification", "")
            if cls not in allowlist:
                return _null_result("classification_mismatch"), None

    # Step 1: load and correct raw frame
    _, frame_n, _ = _parse_label_parts(label)
    frame_jpg = chip_dir / "scan_10x" / f"frame_{frame_n:04d}.jpg"
    if not frame_jpg.exists():
        raise FileNotFoundError(f"Raw frame not found: {frame_jpg} (label '{label}')")

    raw = cv2.imread(str(frame_jpg))
    if raw is None:
        raise FileNotFoundError(f"cv2.imread returned None for: {frame_jpg}")
    corrected = apply_flatfield(raw, flatfield)

    W, H = get_frame_dims(meta)
    pixel_um = get_pixel_um(meta)

    # Step 2: build flake mask + dilated mask
    mask_full = build_flake_mask(det, H, W)

    kernel_px = _odd_round(VARIANCE_KERNEL_UM / pixel_um)

    # Small-flake guard
    size_px = det.get("size_px", 0)
    if size_px > 0 and size_px < kernel_px ** 2:
        import warnings
        warnings.warn(
            f"Flake '{label}' size_px={size_px} < kernel_px^2={kernel_px**2}; "
            "reducing kernel to 3px",
            RuntimeWarning,
            stacklevel=4,
        )
        kernel_px = 3

    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_dilated = cv2.dilate(mask_full.astype(np.uint8), disk).astype(bool)

    # Step 3: crop to bbox + padding
    bbox = det.get("bbox")
    if bbox is None:
        # compute from mask if bbox missing
        ys, xs = np.where(mask_full)
        if len(xs) == 0:
            return _null_result("empty_mask"), None
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1)]

    bx, by, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    pad = max(CROP_PAD_PX, kernel_px + 2)
    row0 = max(0, by - pad)
    row1 = min(H, by + bh + pad)
    col0 = max(0, bx - pad)
    col1 = min(W, bx + bw + pad)

    green_crop = corrected[row0:row1, col0:col1, 1].astype(np.float32)
    mask_crop = mask_full[row0:row1, col0:col1]
    mask_d_crop = mask_dilated[row0:row1, col0:col1]

    # Step 4: local variance map on green channel (E[X²] - E[X]²)
    k = kernel_px
    box_kernel = np.ones((k, k), dtype=np.float32) / (k * k)
    mean_map = cv2.filter2D(green_crop, -1, box_kernel)
    mean_sq_map = cv2.filter2D(green_crop ** 2, -1, box_kernel)
    var_map = np.clip(mean_sq_map - mean_map ** 2, 0.0, None)

    # Step 5: adaptive threshold
    # Variance is computed over the dilated crop (so border pixels get full
    # neighbourhood context), but the percentile is sampled from the strict
    # interior only. Dilation buffer pixels have elevated variance by construction
    # (they straddle the flake edge), so including them would inflate the
    # threshold and let genuine edge noise through.
    in_mask_var = var_map[mask_crop]
    if in_mask_var.size == 0:
        return _null_result("empty_mask"), None

    pct = variance_percentile if variance_percentile is not None else VARIANCE_PERCENTILE
    threshold = float(np.percentile(in_mask_var, pct))
    clean_mask = (var_map < threshold) & mask_crop

    # Step 6: largest axis-aligned rectangle in clean_mask
    rect = max_rect_in_binary_mask(clean_mask)
    if rect is None or rect[2] * rect[3] == 0:
        return _null_result("no_clean_region"), None

    rx_c, ry_c, rw, rh = rect  # crop-space coordinates

    # Step 7: convert to frame space and stage space
    lhrr_bbox_frame = (col0 + rx_c, row0 + ry_c, rw, rh)
    lhrr_bbox_stage = pixel_bbox_to_stage(lhrr_bbox_frame, frame_entry, pixel_um, W, H)

    lhrr_area_px = rw * rh
    lhrr_area_um2 = round(lhrr_area_px * pixel_um ** 2, 3)
    flake_size_px = det.get("size_px") or max(1, int(mask_full.sum()))
    lhrr_fraction = round(lhrr_area_px / flake_size_px, 4)
    lhrr_side_um = lhrr_area_um2 ** 0.5
    useful_threshold_um = _useful_threshold_um(material)
    lhrr_quality_flag = "usable" if lhrr_side_um >= useful_threshold_um else "marginal"

    result: dict[str, Any] = {
        "lhrr_area_px": lhrr_area_px,
        "lhrr_area_um2": lhrr_area_um2,
        "lhrr_fraction": lhrr_fraction,
        "lhrr_bbox_frame": lhrr_bbox_frame,
        "lhrr_bbox_stage": lhrr_bbox_stage,
        "variance_threshold_used": round(threshold, 4),
        "kernel_size_px": kernel_px,
        "lhrr_quality_flag": lhrr_quality_flag,
        "lhrr_skip_reason": None,
    }

    if not diagnostics:
        return result, None

    diag: dict[str, Any] = {
        "corrected_crop": corrected[row0:row1, col0:col1],
        "var_map": var_map,
        "mask_crop": mask_crop,
        "mask_d_crop": mask_d_crop,
        "clean_mask": clean_mask,
        "rect_crop": (rx_c, ry_c, rw, rh),
    }
    return result, diag


def _parse_label_parts(label: str) -> tuple[int, int, int]:
    """Extract (rank, frame_n, d_id) from label without re-importing regex."""
    import re
    m = re.match(r"^rank(\d+)_frame_(\d+)_d(\d+)$", label)
    if not m:
        raise ValueError(f"Bad label: '{label}'")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _save_figure(
    label: str,
    result: dict[str, Any],
    diag: dict[str, Any] | None,
    out_path: Path,
) -> None:
    """Save 4-panel diagnostic PNG."""
    if diag is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        reason = result.get("lhrr_skip_reason", "unknown")
        ax.text(0.5, 0.5, f"Skipped: {reason}", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title(label)
        ax.axis("off")
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    corrected_crop = diag["corrected_crop"]
    var_map = diag["var_map"]
    mask_crop = diag["mask_crop"]
    mask_d_crop = diag["mask_d_crop"]
    clean_mask = diag["clean_mask"]
    rx_c, ry_c, rw, rh = diag["rect_crop"]

    # BGR -> RGB for display
    crop_rgb = corrected_crop[:, :, ::-1]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(label, fontsize=11)

    # Panel 1: corrected crop + mask outline
    ax = axes[0]
    ax.imshow(crop_rgb)
    _overlay_mask_outline(ax, mask_crop, color="red")
    ax.set_title("Corrected + mask outline")
    ax.axis("off")

    # Panel 2: variance heatmap (NaN outside dilated mask → dark gray)
    ax = axes[1]
    var_display = var_map.copy()
    var_display[~mask_d_crop] = np.nan
    cmap = plt.cm.hot.copy()
    cmap.set_bad(color="#333333")
    ax.imshow(var_display, cmap=cmap, interpolation="nearest")
    ax.set_title(f"Variance (thresh={result['variance_threshold_used']:.1f})")
    ax.axis("off")

    # Panel 3: binary clean mask
    ax = axes[2]
    display = np.zeros((*mask_crop.shape, 3), dtype=np.uint8)
    # green = clean inside mask, dark red = dirty inside mask, black = outside
    display[mask_crop & clean_mask] = [0, 200, 0]
    display[mask_crop & ~clean_mask] = [120, 0, 0]
    ax.imshow(display)
    ax.set_title("Clean mask (green=clean, dark-red=dirty)")
    ax.axis("off")

    # Panel 4: corrected crop + LHRR bbox + mask outline
    ax = axes[3]
    ax.imshow(crop_rgb)
    _overlay_mask_outline(ax, mask_crop, color="red")
    import matplotlib.patches as mpatches
    rect_patch = mpatches.Rectangle(
        (rx_c, ry_c), rw, rh,
        linewidth=2, edgecolor="lime", facecolor="none"
    )
    ax.add_patch(rect_patch)
    area_um2 = result["lhrr_area_um2"]
    frac = result["lhrr_fraction"]
    ax.set_title(f"LHRR {area_um2:.0f} µm²  frac={frac:.2f}")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _overlay_mask_outline(ax: Any, mask: np.ndarray, color: str) -> None:
    """Draw the contour of a binary mask on a matplotlib axes."""
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        pts = cnt[:, 0, :]
        ax.plot(
            np.append(pts[:, 0], pts[0, 0]),
            np.append(pts[:, 1], pts[0, 1]),
            color=color, linewidth=1.5,
        )
