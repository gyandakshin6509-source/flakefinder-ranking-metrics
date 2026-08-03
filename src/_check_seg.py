"""
_check_seg.py: quick schema probe for a chip's seg/ directory.

Prints the count of seg JSONs vs. scan_10x JPGs and dumps the field set of
one sample detection. Used during development to discover the seg detection
schema (classification, thickness_nm, etc.); kept as a generic debugging
tool.

Usage:
    python src/_check_seg.py <chip_dir> [<seg_frame_index>]

  chip_dir          path to a chip_N directory (containing seg/ and scan_10x/)
  seg_frame_index   optional index into the sorted seg JSON list; defaults to
                    the middle file
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(f"Usage: python {Path(sys.argv[0]).name} <chip_dir> [<seg_frame_index>]")
        sys.exit(2)
    chip = Path(sys.argv[1])
    if not chip.is_dir():
        print(f"ERROR: chip_dir not found or not a directory: {chip}")
        sys.exit(1)

    seg = sorted(chip.glob("seg/frame_*.json"))
    jpg = sorted(chip.glob("scan_10x/frame_*.jpg"))
    if not seg:
        print(f"ERROR: no seg JSONs under {chip / 'seg'}")
        sys.exit(1)

    idx = int(sys.argv[2]) if len(sys.argv) == 3 else len(seg) // 2
    if not 0 <= idx < len(seg):
        print(f"ERROR: seg_frame_index {idx} out of range [0, {len(seg)})")
        sys.exit(1)

    print(f"chip: {chip}")
    print(f"seg JSONs: {len(seg)}  range: {seg[0].stem} .. {seg[-1].stem}")
    print(f"jpg files: {len(jpg)}  range: "
          f"{jpg[0].stem if jpg else '-'} .. {jpg[-1].stem if jpg else '-'}")

    sample = json.loads(seg[idx].read_text())
    dets = sample.get("detections", [])
    print(f"\nsample seg JSON: {seg[idx].name}")
    print(f"sample frame n_detections: {len(dets)}")
    if not dets:
        return

    d = dets[0]
    print(f"  all keys: {list(d.keys())}")
    truncated = {
        k: (v if not isinstance(v, list) or len(v) <= 6 else f"<list len={len(v)}>")
        for k, v in d.items()
    }
    print(f"  full det: {json.dumps(truncated, indent=2)}")


if __name__ == "__main__":
    main()
