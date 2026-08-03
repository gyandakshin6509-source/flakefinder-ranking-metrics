"""
io_utils.py: shared I/O utilities for FlakeFinder metrics.

Provides:
  - parse_revisit_label   label -> (rank, frame_n, d_id)
  - load_scan_meta        path  -> dict  (LRU-cached)
  - lookup_detection      chip_dir + label -> (det, frame_entry, meta)
  - apply_flatfield       raw BGR + flatfield -> corrected BGR
  - build_flake_mask      detection + dims -> bool mask
  - project_contour_to_stage  pixel contour -> stage-coord contour
  - pixel_bbox_to_stage   pixel bbox -> stage bbox (µm)
  - get_frame_dims        meta -> (W, H)
  - get_pixel_um          meta -> float
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r"^rank(\d+)_frame_(\d+)_d(\d+)$")


def parse_revisit_label(label: str) -> tuple[int, int, int]:
    """
    Parse a revisit label into (rank, frame_n, d_id).

    d_id is the 0-based direct index into the frame's detections[] array,
    verified empirically: detections[d_id] gives the centroid closest to the
    stage position recorded in revisit_50x.json (residuals 86-143 µm,
    consistent with parcentric shift at 50x).

    Parameters
    ----------
    label : str
        Candidate label, e.g. 'rank01_frame_0432_d1'.

    Returns
    -------
    (rank, frame_n, d_id) as ints.

    Raises
    ------
    ValueError if the label does not match the expected format.
    """
    m = _LABEL_RE.match(label)
    if not m:
        raise ValueError(
            f"Revisit label '{label}' does not match expected format "
            "'rank{NN}_frame_{NNNN}_d{N}'"
        )
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


# ---------------------------------------------------------------------------
# scan_meta loading (cached)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def load_scan_meta(path: str) -> dict[str, Any]:
    """
    Load and cache a scan_meta.json file.

    Caches by string path for lru_cache hashability. Call with str(path).

    Parameters
    ----------
    path : str
        Absolute path to scan_meta.json.

    Returns
    -------
    Parsed JSON dict.

    Raises
    ------
    FileNotFoundError if the file is absent.
    KeyError if any critical top-level field is missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"scan_meta.json not found: {p}")
    meta = json.loads(p.read_text())
    for field in ("frames", "optics", "camera",
                  "x_min_um", "x_max_um", "y_min_um", "y_max_um"):
        if field not in meta:
            raise KeyError(f"Expected field '{field}' missing from {p}")
    return meta


def get_frame_dims(meta: dict[str, Any]) -> tuple[int, int]:
    """
    Return (width_px, height_px) from scan_meta camera sub-dict.

    Parameters
    ----------
    meta : dict
        Parsed scan_meta.json.

    Returns
    -------
    (W, H) as ints.
    """
    cam = meta["camera"]
    return int(cam["frame_width_px"]), int(cam["frame_height_px"])


def get_pixel_um(meta: dict[str, Any]) -> float:
    """
    Return the sample-plane pixel size in µm from scan_meta optics.

    Parameters
    ----------
    meta : dict
        Parsed scan_meta.json.

    Returns
    -------
    Sample-plane pixel pitch in µm (sample_pixel_x_um).
    """
    return float(meta["optics"]["sample_pixel_x_um"])


def _frames_by_n(meta: dict[str, Any]) -> dict[int, dict]:
    """Build {frame_n: frame_entry} lookup from scan_meta["frames"]."""
    return {f["n"]: f for f in meta["frames"]}


# ---------------------------------------------------------------------------
# Detection lookup
# ---------------------------------------------------------------------------

def lookup_detection(
    chip_dir: Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Resolve a revisit label to its detection, frame metadata, and scan_meta.

    Parameters
    ----------
    chip_dir : Path
        Chip root directory (contains scan_10x/ and seg/).
    label : str
        Revisit candidate label, e.g. 'rank01_frame_0432_d1'.

    Returns
    -------
    detection : dict
        Full detection dict from chip_dir/seg/frame_NNNN.json.
    frame_entry : dict
        Entry from scan_meta["frames"] with x_um, y_um, phase, etc.
    scan_meta : dict
        Full parsed scan_meta (for camera, optics, coordinate bounds).

    Raises
    ------
    ValueError: label does not match regex.
    FileNotFoundError: seg JSON or scan_meta.json missing.
    KeyError: 'detections' or 'contour' field absent.
    IndexError: d_id out of range for that frame.
    ValueError: contour has fewer than 3 points.
    """
    _, frame_n, d_id = parse_revisit_label(label)

    seg_file = chip_dir / "seg" / f"frame_{frame_n:04d}.json"
    if not seg_file.exists():
        raise FileNotFoundError(
            f"Seg JSON not found for label '{label}': {seg_file}"
        )

    seg_data = json.loads(seg_file.read_text())
    detections = seg_data.get("detections")
    if detections is None:
        raise KeyError(
            f"'detections' field missing from {seg_file} (label '{label}')"
        )

    if d_id >= len(detections):
        raise IndexError(
            f"d_id={d_id} out of range: frame has {len(detections)} detections "
            f"(label '{label}', file: {seg_file})"
        )

    det = detections[d_id]

    if "contour" not in det:
        raise KeyError(
            f"'contour' field missing from detection d_id={d_id} "
            f"in {seg_file} (label '{label}')"
        )
    if len(det["contour"]) < 3:
        raise ValueError(
            f"Contour has {len(det['contour'])} points, too small to form a mask "
            f"(label '{label}')"
        )

    meta_path = chip_dir / "scan_10x" / "scan_meta.json"
    meta = load_scan_meta(str(meta_path))

    by_n = _frames_by_n(meta)
    if frame_n not in by_n:
        raise KeyError(
            f"Frame n={frame_n} not found in scan_meta.json frames array "
            f"(label '{label}', chip: {chip_dir.name})"
        )

    return det, by_n[frame_n], meta


# ---------------------------------------------------------------------------
# Flatfield correction
# ---------------------------------------------------------------------------

def apply_flatfield(raw: np.ndarray, flatfield: np.ndarray) -> np.ndarray:
    """
    Apply multiplicative flatfield correction to a raw BGR uint8 frame.

    corrected = clip(raw * flatfield, 0, 255).astype(uint8)

    Parameters
    ----------
    raw : np.ndarray
        Raw frame, shape (H, W, 3), dtype uint8, BGR channel order.
    flatfield : np.ndarray
        Correction factors, shape (H, W, 3), dtype float32, BGR channel order.
        Values centred near 1.0 (>1 brightens, <1 darkens).

    Returns
    -------
    Flatfield-corrected uint8 BGR frame.
    """
    return (raw.astype(np.float32) * flatfield).clip(0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def build_flake_mask(det: dict[str, Any], H: int, W: int) -> np.ndarray:
    """
    Build a binary bool mask for a detection contour in a frame of size H×W.

    Uses cv2.fillPoly on det["contour"] which stores [[col, row], ...] pixel
    coordinates in the local frame.

    Parameters
    ----------
    det : dict
        Detection dict containing a "contour" field.
    H, W : int
        Frame height and width in pixels.

    Returns
    -------
    bool ndarray of shape (H, W).
    """
    contour = np.array(det["contour"], dtype=np.int32)  # (N, 2): col, row
    canvas = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(canvas, [contour], 255)
    return canvas.astype(bool)


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def project_contour_to_stage(
    contour: list[list[float]] | np.ndarray,
    frame_entry: dict[str, Any],
    pixel_um: float,
    W: int,
    H: int,
) -> np.ndarray:
    """
    Project a pixel-space contour [[col, row], ...] to stage coordinates (µm).

    Transform:
      stage_x = frame_x_um + (col - W/2) * pixel_um
      stage_y = frame_y_um + (row - H/2) * pixel_um

    Parameters
    ----------
    contour : array-like, shape (N, 2)
        Pixel coordinates [[col, row], ...].
    frame_entry : dict
        Frame metadata entry with 'x_um' and 'y_um' (stage position of frame centre).
    pixel_um : float
        Sample-plane pixel pitch in µm.
    W, H : int
        Frame width and height in pixels.

    Returns
    -------
    float32 ndarray of shape (N, 2) with stage coordinates (x_um, y_um).
    """
    pts = np.asarray(contour, dtype=np.float32)
    out = np.empty_like(pts)
    out[:, 0] = frame_entry["x_um"] + (pts[:, 0] - W / 2.0) * pixel_um
    out[:, 1] = frame_entry["y_um"] + (pts[:, 1] - H / 2.0) * pixel_um
    return out


def pixel_bbox_to_stage(
    bbox_frame: tuple[int, int, int, int],
    frame_entry: dict[str, Any],
    pixel_um: float,
    W: int,
    H: int,
) -> tuple[float, float, float, float]:
    """
    Convert a pixel-space axis-aligned bbox to stage coordinates.

    Parameters
    ----------
    bbox_frame : (col_left, row_top, width_px, height_px)
        Bounding box in frame pixel coordinates.
    frame_entry : dict
        Frame metadata entry with 'x_um' and 'y_um'.
    pixel_um : float
        Sample-plane pixel pitch in µm.
    W, H : int
        Frame width and height in pixels.

    Returns
    -------
    (stage_x, stage_y, stage_w_um, stage_h_um) where stage_x/y is the
    top-left corner in stage µm.
    """
    col, row, w_px, h_px = bbox_frame
    stage_x = frame_entry["x_um"] + (col - W / 2.0) * pixel_um
    stage_y = frame_entry["y_um"] + (row - H / 2.0) * pixel_um
    return (
        round(stage_x, 3),
        round(stage_y, 3),
        round(w_px * pixel_um, 3),
        round(h_px * pixel_um, 3),
    )
