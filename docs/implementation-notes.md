# Implementation notes

Design decisions that were made *during* implementation rather than in the
original plan, with the evidence behind each. `docs/design-plan.md` is the
pre-implementation plan; where the two disagree, this file is current.

---

## Dataset the metrics were developed against

| Run | Material preset | Chips | Candidates |
|---|---|---|---|
| `run_20260505_1616` (hBN) | `hbn_medium` | 6 | 88 |
| `run_20260504_1026` (graphene) | `graphene_thin_90nm` | 2 | 147 |

235 candidates total. The raw scan archives are lab data and are not part of
this repository.

## Pixel scale (10x)

Leica K5C physical pixel: 2.4 µm. 3x3 binning gives a 7.2 µm effective pitch.
At the 10x objective that is **0.72 µm/pixel** at the sample plane, confirmed
against `optics.sample_pixel_x_um` in `scan_meta.json`.

---

## LHRR

### `VARIANCE_PERCENTILE` default: 50 (plan said 25)

A sensitivity sweep at p25 / p50 / p75 on three chip_0 hBN flakes
(small 505 µm², median 978 µm², large 4556 µm²):

- **p25** — clean mask fragments; LHRR undershoots the visually usable area
  (fractions 4–8%)
- **p50** — clean mask contiguous on all three; LHRR lands in the flat
  interior (fractions 10–20%) ← **chosen**
- **p75** — threshold starts accepting genuine edge variance; on the large
  flake the LHRR box extends into the ragged border

Re-run the sweep (`src/sweep_lhrr_percentile.py`) if a new material preset or
illumination condition is introduced.

### Variance threshold is sampled from the strict interior

The variance map is computed over the **dilated** mask crop so border pixels
receive a full box-filter neighbourhood. The percentile threshold, however, is
sampled from `var_map[mask_crop]` — the strict flake interior, no dilation
buffer.

Buffer pixels straddle the flake edge and carry ~2.8x the mean variance of the
strict interior (measured on the small flake: buffer 604.6 vs interior 217.4).
Including them shifts the 25th percentile by +6.3% on smooth-edge flakes and
−6.9% on high-contrast-edge flakes (large hBN: 211 fewer clean pixels).
Sampling from the interior gives a threshold calibrated to interior surface
quality only.

`clean_mask = (var_map < threshold) & mask_crop` — the strict boundary is
always applied to the output mask regardless of where the threshold came from.

### Per-material `usable` / `marginal` cutoff

`lhrr_quality_flag` is derived per material, not from one global constant:

```python
LHRR_USEFUL_THRESHOLD_UM_BY_MATERIAL: dict[str, float] = {
    "hbn_medium":         20.0,
    "graphene_thin_90nm": 10.0,
}
DEFAULT_LHRR_USEFUL_THRESHOLD_UM: float = 15.0   # fallback, prints a warning
```

With the original global 20 µm threshold, 146 of 147 graphene candidates
(>99%) flagged `"marginal"` — the flag had stopped differentiating the
population. At 10 µm graphene yields 36 usable / 111 marginal, matching the
visually distinct subset. hBN's 20 µm is unchanged. Both are conservative
starting points, not calibrated to any particular stamp tolerance.

The flag is a label only. Composite ranking always uses `lhrr_area_um2`
(run-normalised), so changing a threshold relabels candidates without
reordering them.

---

## EOP

### `weighted_obstruction` is an area integral, not a pixel sum

```
weighted_obstruction = Σ_pixels (obs[j] / d_j²) * stage_px²
```

The `stage_px²` factor (pixel area in µm²) converts the discrete pixel sum into
a proper area integral, giving units of
`(normalised-contrast / µm²) * µm²` — dimensionless. Without it the sum scales
as `1/stage_px²`, so a finer occupancy resolution would inflate the obstruction
value purely from increased pixel count and `eop_score` would not be comparable
across `OCCUPANCY_STAGE_PX_UM` settings.

Verified on the three validation flakes at 5 / 2 / 1 µm/px: post-fix drift is
~1–4% for non-trivial cases, versus ~25% pre-fix. See
`src/eop.py:compute_eop` (search for `stage_px ** 2`).

### Clearance pins to `OCCUPANCY_STAGE_PX_UM` for adjacent flakes

For candidates whose seg contour touches another detection, measured clearance
is exactly one occupancy pixel (5 µm at the default resolution). This is
structural — sub-pixel separation is not measurable on the occupancy grid — not
a bug. `clearance_warning` (`clearance_um < EOP_MIN_CLEARANCE_UM`) is the right
way to identify these cases; the absolute clearance value of a warned candidate
is not informative on its own.

113 of 235 candidates (48%) sit at the resolution floor on this dataset.

---

## QC status

A 10-check QC harness (`src/qc_run.py`) was run before handoff:

- **Deterministic.** A full re-run of the graphene pipeline produced a
  byte-identical `ranked_candidates.csv` (0 differing rows, max numeric
  difference 0.0).
- **235 candidates across 8 chips** processed.
- **Schema, range, and completeness checks PASS.** Every CSV row carries the
  documented field set; every numeric field is in range; per-chip
  `revisit_50x.json` point count equals JSON candidate count equals CSV row
  count for all 8 chips.
- Two ship-blockers were resolved: `requirements.txt` added with pinned
  versions, and `src/_check_seg.py` changed to take a chip directory as a CLI
  argument instead of a hardcoded absolute path.

---

## Code standards used here

- Python 3.10+, Windows-compatible paths
- Type hints throughout; constants at the top of each module
- Every public function has a docstring with params, returns, and a one-line
  summary
- Missing JSON fields fail loudly (`KeyError`) — never a silent skip
- No deep learning. Dependencies limited to numpy, opencv-python, scipy,
  pandas, matplotlib
- Per-chip progress printed to stdout
