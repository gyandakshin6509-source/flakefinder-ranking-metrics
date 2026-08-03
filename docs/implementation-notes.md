# Implementation notes

These are the decisions I made while actually building the metrics, rather than
the ones I planned up front, along with the numbers I based them on.
`design-plan.md` is the pre-implementation plan. Where the two disagree, this
file is the current one.

---

## The dataset these were developed on

| Run | Material preset | Chips | Candidates |
|---|---|---|---|
| `run_20260505_1616` (hBN) | `hbn_medium` | 6 | 88 |
| `run_20260504_1026` (graphene) | `graphene_thin_90nm` | 2 | 147 |

235 candidates in total. The raw scan archives belong to the lab and aren't
part of this repo.

## Pixel scale at 10x

The Leica K5C's physical pixel is 2.4 µm, and 3x3 binning takes that to a
7.2 µm effective pitch. Through the 10x objective that comes out to
**0.72 µm/pixel** at the sample plane. I checked this against
`optics.sample_pixel_x_um` in `scan_meta.json` rather than trusting the
arithmetic on its own.

---

## LHRR

### `VARIANCE_PERCENTILE` defaults to 50, not 25 like the plan said

I swept the threshold at the 25th, 50th and 75th percentile on three chip_0 hBN
flakes: a small one at 505 µm², a median one at 978 µm², and a large one at
4556 µm².

- **p25**: the clean mask fragments, and LHRR ends up well under the area that
  looks usable by eye. Fractions land around 4 to 8%.
- **p50**: clean mask stays contiguous on all three, and the rectangle sits in
  the flat interior where you'd want it. Fractions around 10 to 20%. This is
  what I went with.
- **p75**: the threshold gets loose enough to accept real edge variance. On the
  large flake the rectangle pushes out into the ragged border.

If a new material preset or a different illumination setup shows up, the sweep
is worth re-running. `src/sweep_lhrr_percentile.py` does it.

### The threshold is sampled from the strict interior

The variance map itself is computed over the dilated mask crop, so that pixels
near the border still get a full box-filter neighbourhood. The percentile
threshold though is sampled from `var_map[mask_crop]`, meaning the strict flake
interior with no dilation buffer at all.

The reason is that buffer pixels straddle the flake edge, and on the small
flake they measured about 2.8 times the mean variance of the interior (604.6
against 217.4). Including them moves the 25th percentile by +6.3% on flakes
with smooth edges and by -6.9% on flakes with high-contrast edges. On the large
hBN flake that second case cost 211 clean pixels, which is not nothing.
Sampling from the interior gives a threshold that describes interior surface
quality and only that.

The output mask is still `(var_map < threshold) & mask_crop`, so the strict
boundary always gets applied regardless of where the threshold came from.

### Usable vs marginal is a per-material cutoff

`lhrr_quality_flag` comes from a per-material dict instead of one global
constant:

```python
LHRR_USEFUL_THRESHOLD_UM_BY_MATERIAL: dict[str, float] = {
    "hbn_medium":         20.0,
    "graphene_thin_90nm": 10.0,
}
DEFAULT_LHRR_USEFUL_THRESHOLD_UM: float = 15.0   # fallback, prints a warning
```

This changed because the original global 20 µm threshold flagged 146 out of 147
graphene candidates as `"marginal"`. At over 99% the flag had stopped telling
anyone anything. Dropping graphene to 10 µm gives 36 usable and 111 marginal,
which lines up with the subset that looks distinct in the figures. hBN stayed
at 20 µm. Both numbers are conservative and neither is calibrated to a specific
stamp tolerance.

The flag is only a label. Composite ranking always uses `lhrr_area_um2`
normalised across the run, so moving a threshold relabels candidates without
changing their order.

---

## EOP

### `weighted_obstruction` is an area integral, not a pixel sum

```
weighted_obstruction = Σ_pixels (obs[j] / d_j²) * stage_px²
```

That `stage_px²` factor is the pixel area in µm², and it's what turns the
discrete pixel sum into a proper area integral. Units work out to
`(normalised-contrast / µm²) * µm²`, so dimensionless. Without it the sum
scales as `1/stage_px²`, which means running the occupancy map at a finer
resolution inflates the obstruction value purely because there are more pixels
in it, and `eop_score` stops being comparable between different
`OCCUPANCY_STAGE_PX_UM` settings.

I caught this by running the three validation flakes at 5, 2 and 1 µm/px and
seeing the value move about 25%. After the fix the drift is around 1 to 4% for
the cases where obstruction isn't near zero anyway. The code is in
`src/eop.py:compute_eop`, search for `stage_px ** 2`.

### Clearance pins to `OCCUPANCY_STAGE_PX_UM` when flakes touch

If a candidate's seg contour is up against another detection, the measured
clearance comes back as exactly one occupancy pixel, so 5 µm at the default
resolution. That's structural rather than a bug, since sub-pixel separation
just isn't measurable on the occupancy grid. `clearance_warning`
(`clearance_um < EOP_MIN_CLEARANCE_UM`) is the right way to find these, and the
absolute clearance number on a warned candidate shouldn't be read on its own.

On this dataset 113 of the 235 candidates, so 48%, sit at that floor.

---

## QC status

`src/qc_run.py` is a 10-check harness that was run before handoff. What it
found:

- **The pipeline is deterministic.** A full re-run of the graphene pipeline
  produced a byte-identical `ranked_candidates.csv`, 0 differing rows, maximum
  numeric difference of 0.0.
- **235 candidates across 8 chips** all processed.
- **Schema, range and completeness checks pass.** Every CSV row has the
  documented field set, every numeric field falls in its expected range, and
  for all 8 chips the `revisit_50x.json` point count matches the JSON candidate
  count matches the CSV row count.
- Two things it flagged as ship-blockers got fixed: `requirements.txt` was
  missing, and `src/_check_seg.py` had a hardcoded absolute path in it instead
  of taking a chip directory as an argument.

---

## Code standards I stuck to

- Python 3.10+, paths that work on Windows
- Type hints throughout, constants at the top of each module
- Docstring on every public function with params, returns and a one-line summary
- Missing JSON fields raise `KeyError` rather than getting skipped quietly
- No deep learning. Dependencies are numpy, opencv-python, scipy, pandas and
  matplotlib
- Per-chip progress printed to stdout so a long run isn't silent
