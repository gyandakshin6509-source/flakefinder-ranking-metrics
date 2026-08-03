"""
Phase 1 reconnaissance script.

Extracts the two run archives, walks the directory tree, inspects JSON schemas,
loads one raw frame + flatfield, generates flatfield_check.png, and writes
outputs/recon/recon.md.

Usage:
    python src/recon.py
"""

from __future__ import annotations

import json
import sys
import tarfile
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zstandard as zstd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "flakes data" / "flakes data"
FLATFIELDS_DIR = DATA_DIR / "flatfields"
OUTPUTS_RECON = ROOT / "outputs" / "recon"
OUTPUTS_RECON.mkdir(parents=True, exist_ok=True)

ARCHIVE_HBN = DATA_DIR / "SF121_D-J_run_20260505_1616.tar.zst"
ARCHIVE_GR  = DATA_DIR / "SF_Gr-260504_run_20260504_1026.tar.zst"

FLATFIELD_10X_NPY  = FLATFIELDS_DIR / "flatfield_10x_bin3.npy"
FLATFIELD_10X_JSON = FLATFIELDS_DIR / "flatfield_10x_bin3.json"

# How many files to show per directory before summarising
DIR_FILE_LIMIT = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_archive(archive: Path, dest: Path) -> None:
    """Extract a .tar.zst archive to dest/."""
    print(f"  Extracting {archive.name} -> {dest} ...", flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    dctx = zstd.ZstdDecompressor()
    with open(archive, "rb") as fh:
        stream = dctx.stream_reader(fh)
        with tarfile.open(fileobj=stream, mode="r|") as tf:
            tf.extractall(dest, filter="data")
    print("  Done.", flush=True)


def tree_counts(root: Path, file_limit: int = DIR_FILE_LIMIT) -> tuple[str, dict[str, int]]:
    """
    Return a concise directory tree and a {ext: count} dict.

    Files within a directory are shown up to file_limit; further files are
    summarised as '... N more files'.
    """
    lines: list[str] = []
    ext_counts: dict[str, int] = defaultdict(int)

    def _walk(p: Path, prefix: str, depth: int) -> None:
        if depth > 5:
            return
        try:
            entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        except PermissionError:
            return

        dirs  = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]

        shown = dirs + files[:file_limit]
        hidden_count = max(0, len(files) - file_limit)

        all_shown = shown + ([] if hidden_count == 0 else ["__ellipsis__"])
        for i, entry in enumerate(all_shown):
            is_last = (i == len(all_shown) - 1)
            connector = "└── " if is_last else "├── "
            extension_prefix = "    " if is_last else "│   "

            if entry == "__ellipsis__":
                lines.append(f"{prefix}{connector}... {hidden_count} more files")
            elif isinstance(entry, Path):
                lines.append(f"{prefix}{connector}{entry.name}")
                if entry.is_dir():
                    _walk(entry, prefix + extension_prefix, depth + 1)
                else:
                    ext_counts[entry.suffix.lower()] += 1

        # Also count hidden files
        for f in files[file_limit:]:
            ext_counts[f.suffix.lower()] += 1

    lines.append(root.name + "/")
    _walk(root, "", 1)
    return "\n".join(lines), dict(ext_counts)


def describe_json_schema(obj: Any, indent: int = 0) -> list[str]:
    """Recursively describe a JSON object's schema (field, type, example)."""
    lines: list[str] = []
    pad = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                lines.append(f"{pad}- **{k}** (object):")
                lines.extend(describe_json_schema(v, indent + 1))
            elif isinstance(v, list):
                elem_type = type(v[0]).__name__ if v else "empty"
                lines.append(f"{pad}- **{k}** (array of {elem_type}, len={len(v)}): `{json.dumps(v[:2])[:80]}...`")
            else:
                lines.append(f"{pad}- **{k}** ({type(v).__name__}): `{json.dumps(v)[:80]}`")
    return lines


def find_good_frame(chip_dir: Path, meta: dict) -> tuple[int, Path]:
    """
    Return (frame_index, path) for a mid-scan capture frame with reasonable brightness.
    Falls back to the first capture frame if no bright frame is found.
    """
    frames = meta["frames"]
    capture_frames = [f for f in frames if f.get("phase") == "capture"]
    if not capture_frames:
        capture_frames = frames

    jpg_dir = chip_dir / "scan_10x"
    best_idx = capture_frames[0]["n"]
    best_mean = 0.0

    # Search mid quarter of the run for a frame with actual substrate content
    search_pool = capture_frames[len(capture_frames)//4 : len(capture_frames)//2]
    for f in search_pool:
        jpg = jpg_dir / f"frame_{f['n']:04d}.jpg"
        if jpg.exists():
            img = cv2.imread(str(jpg))
            if img is not None:
                m = float(img[:, :, 1].mean())  # green channel
                if m > best_mean:
                    best_mean = m
                    best_idx = f["n"]

    return best_idx, jpg_dir / f"frame_{best_idx:04d}.jpg"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sections: list[str] = []

    # -----------------------------------------------------------------------
    # Extract archives
    # -----------------------------------------------------------------------
    extract_dir = ROOT / "data" / "_extracted"
    run_hbn_dir = extract_dir / "run_20260505_1616"
    run_gr_dir  = extract_dir / "run_20260504_1026"

    if not run_hbn_dir.exists():
        print("Extracting hBN run ...")
        extract_archive(ARCHIVE_HBN, extract_dir)
    else:
        print(f"hBN run already extracted.")

    if not run_gr_dir.exists():
        print("Extracting graphene run ...")
        extract_archive(ARCHIVE_GR, extract_dir)
    else:
        print(f"Graphene run already extracted.")

    def find_run_dir(base: Path, run_name: str) -> Path:
        candidate = base / run_name
        if candidate.exists():
            return candidate
        for child in base.iterdir():
            deeper = child / run_name
            if deeper.exists():
                return deeper
        for p in base.rglob("checkpoint.json"):
            if run_name in str(p):
                return p.parent
        raise FileNotFoundError(f"Could not locate run dir for {run_name} under {base}")

    run_hbn = find_run_dir(extract_dir, "run_20260505_1616")
    run_gr  = find_run_dir(extract_dir, "run_20260504_1026")
    print(f"hBN run: {run_hbn.name}")
    print(f"Graphene run: {run_gr.name}")

    # -----------------------------------------------------------------------
    # Section 1: Directory tree (condensed)
    # -----------------------------------------------------------------------
    print("Building directory tree ...")
    tree_str, ext_counts = tree_counts(run_hbn)
    sections.append("## 1. Directory tree: hBN run (SF121 D-J)\n")
    sections.append("```")
    sections.append(tree_str)
    sections.append("```\n")
    sections.append("**File counts by extension:**\n")
    for ext, n in sorted(ext_counts.items(), key=lambda x: -x[1]):
        sections.append(f"- `{ext or '(none)'}`: {n}")
    sections.append("")

    # -----------------------------------------------------------------------
    # Section 2: seg/frame_NNNN.json schema
    # -----------------------------------------------------------------------
    print("Inspecting seg frame JSON ...")
    seg_jsons = sorted(run_hbn.rglob("seg/frame_*.json"))
    if not seg_jsons:
        raise FileNotFoundError("No seg/frame_*.json found under " + str(run_hbn))
    seg_example = json.loads(seg_jsons[0].read_text())
    sections.append("## 2. Schema: `seg/frame_NNNN.json`\n")
    sections.append(f"Example file: `{seg_jsons[0].relative_to(run_hbn)}`\n")
    sections.append("Top-level fields:\n")
    sections.extend(describe_json_schema(seg_example))
    if seg_example.get("detections"):
        sections.append("\nFirst detection object:\n")
        sections.extend(describe_json_schema(seg_example["detections"][0], indent=1))
    sections.append(f"\n> Note: `hull` and `contour` are lists of [x, y] pixel coordinates")
    sections.append(f"> in the local frame. Mask reconstruction uses `cv2.fillPoly` on `contour`.")
    sections.append(f"> Frame index key in `scan_meta.json` is **`n`** (not `index`).")
    sections.append("")

    # -----------------------------------------------------------------------
    # Section 3: scan_meta.json
    # -----------------------------------------------------------------------
    print("Inspecting scan_meta.json ...")
    scan_metas = sorted(run_hbn.rglob("scan_10x/scan_meta.json"))
    if not scan_metas:
        raise FileNotFoundError("No scan_10x/scan_meta.json found")
    meta = json.loads(scan_metas[0].read_text())
    sections.append("## 3. Schema: `scan_meta.json`\n")
    sections.append(f"Example: `{scan_metas[0].relative_to(run_hbn)}`\n")
    sections.append("Top-level scalar fields:\n")
    scalar_fields = {k: v for k, v in meta.items() if not isinstance(v, (dict, list))}
    sections.extend(describe_json_schema(scalar_fields))
    sections.append("\nSub-objects:\n")
    for k in ("scan_params", "camera", "optics", "lighting", "focus_plane"):
        if k in meta:
            sections.append(f"**`{k}`**:")
            sections.extend(describe_json_schema(meta[k], indent=1))
            sections.append("")

    optics = meta.get("optics", {})
    pixel_um: float = optics.get("sample_pixel_x_um", 0.7207)
    fw = optics.get("frame_width_um", meta.get("camera", {}).get("frame_width_px", 1824) * pixel_um)
    fh_um = optics.get("frame_height_um", meta.get("camera", {}).get("frame_height_px", 1216) * pixel_um)
    W = meta.get("camera", {}).get("frame_width_px", 1824)
    H = meta.get("camera", {}).get("frame_height_px", 1216)

    sections.append("\n**Coordinate system summary:**")
    sections.append("- Stage X/Y in micrometres (um). Origin top-left of stage. +Y is down.")
    sections.append("- Frame-to-stage transform (from data_format.md):")
    sections.append("  `stage_x_um = frame_x_um + (px_col - W/2) * sample_pixel_x_um`")
    sections.append("  `stage_y_um = frame_y_um + (px_row - H/2) * sample_pixel_y_um`")
    sections.append(f"- `sample_pixel_x_um` = `{pixel_um}` (from optics)")
    sections.append(f"- Frame size: {W} x {H} px  ({fw:.1f} x {fh_um:.1f} um at sample plane)")
    sections.append("")

    if meta.get("frames"):
        fr0 = meta["frames"][0]
        sections.append("**First `frames` array entry (field names):**\n")
        sections.extend(describe_json_schema(fr0, indent=1))
        sections.append(f"\n> Frame index field is **`n`**. Phase values: `lead_in`, `capture`, `lead_out`.")
        sections.append("")

    # -----------------------------------------------------------------------
    # Section 4: revisit_50x.json and summary.json
    # -----------------------------------------------------------------------
    print("Inspecting revisit_50x.json and summary.json ...")
    revisit_jsons = sorted(run_hbn.rglob("seg/revisit_50x.json"))
    summary_jsons = sorted(run_hbn.rglob("seg/summary.json"))

    sections.append("## 4. Schema: `seg/revisit_50x.json` and `seg/summary.json`\n")

    if revisit_jsons:
        rv = json.loads(revisit_jsons[0].read_text())
        sections.append(f"### `seg/revisit_50x.json`: `{revisit_jsons[0].relative_to(run_hbn)}`\n")
        if isinstance(rv, list):
            sections.append(f"Top-level: array of {len(rv)} entries. First entry:\n")
            if rv:
                sections.extend(describe_json_schema(rv[0], indent=1))
        elif isinstance(rv, dict) and "points" in rv:
            sections.extend(describe_json_schema({k: v for k, v in rv.items() if k != "points"}))
            sections.append(f"\n- **points** (array of dict, len={len(rv['points'])}): ranked candidates")
            if rv["points"]:
                sections.append("\n  First points entry:")
                sections.extend(describe_json_schema(rv["points"][0], indent=2))
        else:
            sections.extend(describe_json_schema(rv))
        sections.append("")

    if summary_jsons:
        sm = json.loads(summary_jsons[0].read_text())
        # Omit the huge detections_by_frame sub-object from the schema
        sm_schema = {k: v for k, v in sm.items() if k != "detections_by_frame"}
        sections.append(f"### `seg/summary.json`: `{summary_jsons[0].relative_to(run_hbn)}`\n")
        sections.extend(describe_json_schema(sm_schema))
        if "detections_by_frame" in sm:
            n_frames_with_dets = len(sm["detections_by_frame"])
            sections.append(f"\n- **detections_by_frame** (object): {n_frames_with_dets} frames, "
                            f"mirrors per-frame seg JSON detections (omitted from schema for brevity)")
        sections.append("")

    # -----------------------------------------------------------------------
    # Section 5: JPEG frame dimensions
    # -----------------------------------------------------------------------
    print("Inspecting JPEG frames ...")
    jpgs = sorted(run_hbn.rglob("scan_10x/frame_*.jpg"))
    if not jpgs:
        raise FileNotFoundError("No scan_10x frame JPEGs found")

    # Use a mid-run capture frame for meaningful analysis
    chip0_dir = sorted(d for d in run_hbn.iterdir() if d.is_dir() and d.name.startswith("chip_"))[0]
    meta0 = json.loads((chip0_dir / "scan_10x" / "scan_meta.json").read_text())
    good_idx, good_jpg = find_good_frame(chip0_dir, meta0)
    print(f"  Using frame_{good_idx:04d}.jpg (mid-run capture) for inspection")

    raw_frame = cv2.imread(str(good_jpg))
    if raw_frame is None:
        raise RuntimeError(f"cv2.imread failed on {good_jpg}")
    h_px, w_px, c = raw_frame.shape

    sections.append("## 5. JPEG frame properties\n")
    sections.append(f"- Example file: `{good_jpg.relative_to(run_hbn)}` (mid-run capture frame)")
    sections.append(f"- Dimensions: {w_px} x {h_px} px (W x H)")
    sections.append(f"- Channels: {c} (OpenCV BGR order)")
    sections.append(f"- dtype: `{raw_frame.dtype}`")
    sections.append(f"- Value range in this frame: [{raw_frame.min()}, {raw_frame.max()}]")
    sections.append(f"- Total scan_10x frames in hBN run: {len(jpgs)}")
    sections.append("")

    # -----------------------------------------------------------------------
    # Section 6: Flatfield + flatfield_check.png
    # -----------------------------------------------------------------------
    print("Loading flatfield and generating flatfield_check.png ...")
    ff = np.load(str(FLATFIELD_10X_NPY))
    sections.append("## 6. Flatfield: `flatfield_10x_bin3.npy`\n")
    sections.append(f"- Shape: `{ff.shape}` (H, W, C), channel order BGR (matches OpenCV)")
    sections.append(f"- dtype: `{ff.dtype}`")
    sections.append(f"- Value range: [{ff.min():.4f}, {ff.max():.4f}]")
    sections.append(f"- Mean: {ff.mean():.4f}")
    sections.append(f"- Interpretation: multiplicative correction factors; 1.0 = no change")
    sections.append(f"- Vignetting (from JSON): 18.6%, centre ~{85.5:.1f} DN vs corners ~{69.6:.1f} DN")

    corrected = (raw_frame.astype(np.float32) * ff).clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Raw frame\n{good_jpg.name}", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Flatfield-corrected\n(vignetting removed)", fontsize=11)
    axes[1].axis("off")
    fig.suptitle("Flatfield correction check -- 10x chip scan frame", fontsize=13, fontweight="bold")
    fig.tight_layout()
    check_png = OUTPUTS_RECON / "flatfield_check.png"
    fig.savefig(str(check_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {check_png}")

    h2, w2 = corrected.shape[:2]
    centre_raw  = float(raw_frame[h2//2-20:h2//2+20, w2//2-20:w2//2+20, 1].mean())
    corner_raw  = float(raw_frame[10:50, 10:50, 1].mean())
    centre_corr = float(corrected[h2//2-20:h2//2+20, w2//2-20:w2//2+20, 1].mean())
    corner_corr = float(corrected[10:50, 10:50, 1].mean())
    ratio_before = centre_raw / max(corner_raw, 1)
    ratio_after  = centre_corr / max(corner_corr, 1)

    sections.append(f"\n**Vignetting check (green channel centre/corner ratio):**")
    sections.append(f"- Raw: {ratio_before:.3f}  ->  Corrected: {ratio_after:.3f}")
    ok = ratio_after < ratio_before
    sections.append(f"- {'OK: correction reduces vignetting as expected.' if ok else 'WARNING: ratio did not decrease -- inspect flatfield.'}")
    sections.append(f"- Saved `outputs/recon/flatfield_check.png`")
    sections.append("")

    # -----------------------------------------------------------------------
    # Section 7: Pixel scale
    # -----------------------------------------------------------------------
    sections.append("## 7. Pixel-to-micron scale at 10x\n")
    sections.append(f"- Leica K5C physical pixel pitch: 2.4 um")
    sections.append(f"- 3x3 binning -> effective pitch: 7.2 um")
    sections.append(f"- 10x objective -> sample-plane pixel size: **{pixel_um:.4f} um/px**")
    sections.append(f"- Confirmed by `optics.sample_pixel_x_um` = `{pixel_um}` in scan_meta.json")
    sections.append("")
    sections.append("Kernel / distance translation table:\n")
    sections.append("| Physical size (um) | Kernel size (px): rounded to odd |")
    sections.append("|---|---|")
    for um in [3, 5, 7, 10, 15, 20, 50, 100]:
        px = round(um / pixel_um)
        if px % 2 == 0:
            px += 1
        sections.append(f"| {um} um | {px} px |")
    sections.append("")

    # -----------------------------------------------------------------------
    # Section 8: Dataset statistics
    # -----------------------------------------------------------------------
    print("Collecting dataset statistics ...")
    sections.append("## 8. Dataset statistics\n")

    for run_label, run_dir in [("hBN -- SF121 D-J", run_hbn), ("Graphene -- Gr-260504", run_gr)]:
        chip_dirs = sorted(d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("chip_"))
        total_dets = 0
        n_revisit = 0
        cal_dists: list[float] = []
        contrasts: list[float] = []
        sizes_um2: list[float] = []

        for chip in chip_dirs:
            seg_dir = chip / "seg"
            if seg_dir.exists():
                for seg_f in seg_dir.glob("frame_*.json"):
                    try:
                        obj = json.loads(seg_f.read_text())
                        for det in obj.get("detections", []):
                            total_dets += 1
                            if "cal_dist" in det:
                                cal_dists.append(float(det["cal_dist"]))
                            if "mean_contrast" in det:
                                contrasts.append(float(det["mean_contrast"]))
                            if "size_um2" in det:
                                sizes_um2.append(float(det["size_um2"]))
                    except (json.JSONDecodeError, KeyError):
                        pass
                rv_json = seg_dir / "revisit_50x.json"
                if rv_json.exists():
                    rv = json.loads(rv_json.read_text())
                    if isinstance(rv, list):
                        n_revisit += len(rv)
                    elif isinstance(rv, dict) and "points" in rv:
                        n_revisit += len(rv["points"])

        sections.append(f"### {run_label}")
        sections.append(f"- Chips: {len(chip_dirs)}")
        sections.append(f"- Total detections (all seg frames): {total_dets:,}")
        if cal_dists:
            sections.append(f"- `cal_dist`: min={min(cal_dists):.3f}, p25={float(np.percentile(cal_dists,25)):.3f}, "
                            f"median={float(np.median(cal_dists)):.3f}, p75={float(np.percentile(cal_dists,75)):.3f}, max={max(cal_dists):.3f}")
        if contrasts:
            sections.append(f"- `mean_contrast`: min={min(contrasts):.1f}, median={float(np.median(contrasts)):.1f}, max={max(contrasts):.1f}")
        if sizes_um2:
            sections.append(f"- `size_um2`: min={min(sizes_um2):.0f}, median={float(np.median(sizes_um2)):.0f}, "
                            f"p95={float(np.percentile(sizes_um2,95)):.0f}, max={max(sizes_um2):.0f}")
        sections.append(f"- Candidates in `seg/revisit_50x.json` (50x revisit list): {n_revisit}")
        sections.append("")

    # -----------------------------------------------------------------------
    # Write recon.md
    # -----------------------------------------------------------------------
    header = textwrap.dedent(f"""\
        # FlakeFinder -- Phase 1 Reconnaissance Report

        **Generated by:** `src/recon.py`
        **Dataset:** `./data/`

        ---

    """)
    recon_md = OUTPUTS_RECON / "recon.md"
    recon_md.write_text(header + "\n".join(sections), encoding="utf-8")
    print(f"\nWrote {recon_md}")

    # -----------------------------------------------------------------------
    # Stdout summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("PHASE 1 RECON COMPLETE")
    print("=" * 70)
    print(f"  recon.md           -> outputs/recon/recon.md")
    print(f"  flatfield_check.png-> outputs/recon/flatfield_check.png")
    print(f"  Pixel scale        -> {pixel_um:.4f} um/px at 10x")
    print(f"  Flatfield          -> shape={ff.shape}, dtype={ff.dtype}, "
          f"range=[{ff.min():.3f}, {ff.max():.3f}]")
    print(f"  Frame dims         -> {w_px}x{h_px} px, {c}ch, dtype={raw_frame.dtype}")
    print(f"  Vignetting         -> before={ratio_before:.3f} -> after={ratio_after:.3f} "
          f"(centre/corner, {'reduced' if ratio_after < ratio_before else 'NOT reduced'})")
    print(f"  Frame index key    -> 'n' (not 'index')")
    print(f"  Mask source        -> contour field, [x,y] pixel coords, use cv2.fillPoly")
    print(f"  Revisit JSON path  -> chip_N/seg/revisit_50x.json (NOT chip_N/revisit_50x.json)")
    print("=" * 70)


if __name__ == "__main__":
    main()
