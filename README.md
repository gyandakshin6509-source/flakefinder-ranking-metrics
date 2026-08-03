# FlakeFinder

This adds two new per-candidate scores on top of what already exists as a part of 
FlakeFinder. To develop this, we pulled data from two of the lab's run archives:
the hBN run (SF121 D-J) and the graphene run (Gr-260504). This gave us a pool of 
235 candidates to revisit and analyze across the 8 chips. It already identifies candidate 
flakes and revisits the top ones at 50x. So, these metrics help answer two questions
the existing scoring doesn't: how much of this flake is actually clean enough to 
stack, and can we physically pick it up without clipping a neighbor at the edges?

**LHRR** (Largest Homogeneous Rectangular Region) finds the largest
axis-aligned, defect-free rectangle inside the interior of a potential candidate. 
This matters because the existing global entropy/gradient score penalises a flake
with a dirty edge even when its interior is perfectly usable — a 4000 µm²
flake with a contaminated edge scores worse than a 200 µm² clean flake,
even though the first is what you actually want for stacking.

**EOP** (Ease of Pickup) measures how much clearance a candidate has from
its nearest neighbours, weighted by how prominent those neighbours are. A
flake that's physically touching a thick adjacent detection is
effectively unusable because the stamp can't land cleanly.

A composite score combines both so the final ranked CSV reflects
usable area and pickup feasibility together.

> **What's in this repository.** Source and documentation only. The lab's raw
> scan archives, flatfields, and the generated `outputs/` (ranked CSVs,
> per-chip JSON, occupancy and LHRR figures) are not included. Numbers quoted
> throughout this README come from running the pipeline on the two archives
> described above.
>
> Design rationale and the measurements behind each default live in
> [`docs/implementation-notes.md`](docs/implementation-notes.md); the
> pre-implementation plan is in [`docs/design-plan.md`](docs/design-plan.md).

---

## Requirements

Python 3.10 or later. Install dependencies with:
pip install -r requirements.txt

Before running on new data, open `src/flake_metrics.py` and update
`FLATFIELD_PATH` to point to the 10× flatfield `.npy` file for your
microscope configuration.

---

## Running it
python src/flake_metrics.py path/to/run_dir

`run_dir` is the extracted root of one run archive. Run once per archive:
python src/flake_metrics.py data/_extracted/run_20260505_1616
python src/flake_metrics.py data/_extracted/run_20260504_1026

Each chip takes roughly 80 seconds to build the chip-wide occupancy map,
plus a few seconds per candidate for LHRR. Expect 10–15 minutes per run.
The script prints progress per chip.

---

## Using the results

The main output is `outputs/metrics/<run>_ranked_candidates.csv`, sorted
descending by `composite_score`. A reasonable starting workflow:

1. Filter `clearance_warning == False` to get candidates with ≥50 µm
   clearance — these are the safest immediate pickup targets.
2. Among those, sort by `composite_score` or `lhrr_area_um2` depending on
   whether you care more about overall quality or raw usable area.
3. Use `lhrr_quality_flag == "usable"` as a quick check that the LHRR is
   above the material-specific area threshold.
4. For any candidate of interest, cross-reference the `label` field back
   to the original revisit images and the chip occupancy QC image in
   `outputs/eop/`.

On these two datasets, composite scores ranged from 0.02 to 0.64, the
largest LHRR was 1219 µm² (hBN) and 274 µm² (graphene), and roughly 76%
of candidates triggered `clearance_warning`. If your results fall
significantly outside these ranges, check the flatfield path and verify
the occupancy QC images look spatially coherent.

### Column reference

| field | unit | meaning |
|---|---|---|
| `rank_in_run` | — | 1-based rank by composite score within this run |
| `chip` | — | source chip directory |
| `material` | — | material preset string from `seg/summary.json` |
| `label` | — | revisit label linking back to the seg pipeline |
| `stage_x_um`, `stage_y_um`, `stage_z_um` | µm | 50× stage position |
| `lhrr_area_um2` | µm² | clean rectangle area — the LHRR ranking signal |
| `lhrr_fraction` | — | LHRR area / total flake area (diagnostic only) |
| `lhrr_quality_flag` | — | `"usable"` or `"marginal"` per material threshold |
| `lhrr_skip_reason` | — | non-null if LHRR computation failed |
| `clearance_um` | µm | distance to nearest other detection |
| `weighted_obstruction` | — | proximity-weighted area integral of contrast |
| `eop_score` | µm | `clearance_um / (1 + weighted_obstruction)` |
| `clearance_warning` | — | True if `clearance_um < EOP_MIN_CLEARANCE_UM` |
| `eop_skip_reason` | — | non-null if EOP computation failed |
| `lhrr_area_norm` | [0,1] | `lhrr_area_um2` / run-wide max |
| `eop_score_norm` | [0,1] | `eop_score` / run-wide max |
| `composite_score` | [0,1] | `0.5 × lhrr_area_norm + 0.5 × eop_score_norm` |

`composite_score` is normalised by run-wide maxima. Scores are not
comparable across runs or material presets — use them for ranking within
a run only.

### Chip-wide QC images

`outputs/eop/<run>_<chip>_occupancy.png` shows two panels: the occupancy
map (all detections projected to stage coordinates) and the obstruction map
(detection contrast heat). Check these before trusting EOP scores — you can
see scan tile boundaries, dust streaks, or unusual density patterns that
might be affecting individual candidates.

---

## Tuning

All tunables are module-level constants. Change the value and re-run.

### `src/lhrr.py`

| constant | default | meaning |
|---|---|---|
| `VARIANCE_PERCENTILE` | `50` | Percentile of interior variance used as the clean/dirty threshold. Lower = stricter clean mask. |
| `VARIANCE_KERNEL_UM` | `5.0` | Spatial scale of the variance window. 5 µm ≈ 7 px at 10×. |
| `LHRR_USEFUL_THRESHOLD_UM_BY_MATERIAL` | `{hbn_medium: 20.0, graphene_thin_90nm: 10.0}` | Per-material minimum effective side length (µm) for `"usable"`. Adjust to match your stacking geometry. |
| `DEFAULT_LHRR_USEFUL_THRESHOLD_UM` | `15.0` | Fallback threshold for unrecognised material presets. Prints a warning when used. |
| `LHRR_CLASSIFICATION_ALLOWLIST` | `{material: None}` | Per-material detection class filter. `None` = accept any class. |

The per-material thresholds are conservative starting points, not
calibrated values. Adjust `LHRR_USEFUL_THRESHOLD_UM_BY_MATERIAL` once
you have a sense of the minimum clean rectangle your devices actually need.

### `src/eop.py`

| constant | default | meaning |
|---|---|---|
| `EOP_MIN_CLEARANCE_UM` | `50.0` | `clearance_warning` threshold in µm. Lower if the default 76% warn rate is too broad for your workflow. |
| `OCCUPANCY_STAGE_PX_UM` | `5.0` | Occupancy map resolution. Lowering improves clearance precision but increases memory and runtime. |
| `MAX_OBSTRUCTION_RADIUS_UM` | `500.0` | Radius within which neighbours contribute to `weighted_obstruction`. |
| `BASELINE_SAMPLE_FRAMES` | `20` | Maximum frames used to estimate the substrate green-channel baseline. |

### `src/flake_metrics.py`

| constant | default | meaning |
|---|---|---|
| `COMPOSITE_W_LHRR` | `0.5` | Weight on the LHRR term in composite. |
| `COMPOSITE_W_EOP` | `0.5` | Weight on the EOP term in composite. |

---

## Validation

LHRR and EOP were validated against three size-representative hBN
candidates from chip_0 of the hBN run (505 µm², 978 µm², 4556 µm²),
chosen before implementation. For each candidate, the variance threshold
was tested at three percentile settings (25, 50, 75); the default of 50
was selected because it produced a contiguous clean mask covering the
visually flat interior on all three flakes, while 25 fragmented the clean
region and 75 admitted visible edge contamination.

EOP was verified to be resolution-invariant after applying an area-integral
correction to the weighted-obstruction sum. Before the fix, `weighted_
obstruction` varied ~25% across occupancy resolutions of 5/2/1 µm/px;
after, variation dropped to ~1–4% for candidates with non-trivial
obstruction values.

The pipeline was also verified to be deterministic: re-running on the
graphene archive produced byte-identical output.

These validations cover pipeline correctness. Validating that high EOP
and high LHRR actually predict successful device assembly requires
comparing the rankings against experimental pickup outcomes — that ground
truth sits with the lab.

---

## Limitations

Composite scores are relative to each run. Both normalised terms use run-wide
maxima, so the top candidate in any run scores near 1.0 by construction.
Do not compare composite scores across runs or material presets.

**~76% of candidates trigger `clearance_warning` on these datasets.** The
chips are dense and the 50 µm threshold is conservative. The flag is
informational by default — flagged candidates appear in the ranking and
their EOP score differentiates them. Lower `EOP_MIN_CLEARANCE_UM` if you
want the warning to be more selective.

**`clearance_warning = True` does not mean the composite is unreliable.**
The composite weights LHRR area and EOP equally, so a candidate with an
unusually large clean rectangle can rank highly even when crowded. The
warn flag is the correct way to filter for pickup-safe candidates
independently of the composite.

**`clearance_um = 5.0 µm` means touching, not 5 µm of breathing room.**
When a candidate's contour is adjacent to another detection at sub-pixel
resolution, the measured clearance pins to the occupancy map resolution
floor (5 µm at the default setting). The absolute value is not meaningful
for these candidates; use `clearance_warning` to identify them.

**LHRR is less reliable for very small flakes.** Candidates whose bounding
box approaches the variance kernel size (5 µm ≈ 7 px at 10×) have variance
maps dominated by the kernel footprint rather than real surface texture.
Treat `lhrr_quality_flag` for candidates below ~50 µm² LHRR as advisory.

**Occupancy build is slow.** Reading and flatfield-correcting every 10×
frame per chip takes roughly 80 seconds at the default 5 µm/px resolution.
If a run needs to be reprocessed, only re-running LHRR (not rebuilding
occupancy) is much faster.

---

## How it works

### LHRR

The system first applies a per-pixel multiplicative flatfield correction 
to the raw \(10\times\) JPEG images to eliminate illumination artifacts. 
The flake mask is then reconstructed by rendering the contour geometry 
from the segmentation JSON metadata.To evaluate surface quality, local 
variance is computed on the green channel of the corrected image using a 
standard box-filter identity. The kernel size is parameterized to match 
the physical scale of the system, defaulting to 5µm (approximately
7 pixels based on the Leica K5C sensor scale of 0.72µm per pixel at
10x. Crucially, the variance map is computed over a dilated version 
of the mask to ensure edge pixels receive a full kernel neighborhood. 
However, the statistical threshold for cleanliness is sampled exclusively 
from the strict flake interior. Pixels straddling the flake boundary 
exhibit roughly 2.8x higher mean variance than interior pixels; 
including them in the percentile calculation skews the threshold toward 
edge artifacts rather than true interior surface quality.The final clean 
mask comprises all interior pixels falling below this variance threshold. 
To locate the optimal stacking zone, the largest axis-aligned rectangle 
within this clean mask is extracted. This is achieved by reducing the 
geometry row-by-row to the largest-rectangle-in-histogram problem, which 
resolves efficiently in linear time, O (rows x columns), using a stack-based 
approach.

### EOP

To assess physical pickup constraints, all segmentation detections from 
the individual 10x scan frames are projected from pixel space 
to physical stage coordinates. This transformation utilizes the per-frame 
stage metadata, where the physical coordinate is mapped linearly relative 
to the frame center using the system’s 0.72µm per pixel scale factor. 
These projected detections are then compiled into a chip-wide occupancy map at 
a spatial resolution of 5µm per pixel. To establish a baseline for 
optical contrast, the substrate's green-channel intensity is estimated per chip. 
The system samples up to 20 detection-free frames, masking out all identified 
flakes prior to calculating the mean substrate intensity. The contrast delta of 
each individual detection is then normalized against the chip-wide maximum to 
yield an obstruction weight bounded between 0 and 1. Because optical contrast on 
a 90nm SiO2 substrate is non-monotonic with respect 
to flake thickness due to thin-film interference, this weight serves as a 
visibility-based proxy for obstruction risk rather than a direct topographic 
thickness measurement.Clearance metrics are subsequently extracted. Spatial 
clearance is measured by applying a distance transform to the inverted occupancy 
map after temporarily omitting the candidate flake. This value is sampled precisely 
at the candidate's outer boundary.Finally, a total weighted obstruction score is 
calculated as an area integral over all non-candidate detections within a defined 
maximum radius. This calculation scales inversely with the squared distance to the 
target. By multiplying the summation by the physical pixel area, the final metric 
behaves as a true area integral, ensuring the score remains invariant to changes in 
the underlying occupancy map resolution.
---

## Folder layout

```
flakefinder/
├── README.md
├── requirements.txt
├── docs/
│   ├── design-plan.md        pre-implementation plan
│   └── implementation-notes.md   decisions made during implementation
├── src/
│   ├── flake_metrics.py      main entrypoint
│   ├── run_all_metrics.py    processes both runs in sequence
│   ├── lhrr.py               LHRR metric
│   ├── eop.py                EOP metric
│   ├── io_utils.py           shared I/O utilities
│   ├── recon.py              dataset recon / flatfield check
│   ├── qc_run.py             10-check QC harness
│   ├── validate_lhrr.py      LHRR validation figures
│   ├── validate_eop.py       EOP validation figures
│   ├── sweep_lhrr_percentile.py  variance-percentile sensitivity sweep
│   ├── diag_*.py             diagnostic scripts used during development
│   └── _check_seg.py         debug tool: inspect seg JSON schema for a chip
│                             usage: python src/_check_seg.py <chip_dir>
└── outputs/                  generated at runtime (not in the repo)
    ├── metrics/              ranked CSVs and per-chip JSON results
    └── eop/                  chip-wide occupancy QC images
```
