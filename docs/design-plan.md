# FlakeFinder Metrics — Phase 2 Implementation Plan

> _This document captures the Phase 2 plan as of 2026-05-08. Actual implementation
> may differ in places — see `CLAUDE.md` and `README.md` for current state._

**Status:** Awaiting review before any code is written.
**Prereq scripts run:** `src/recon.py`, `src/_check_plan_prereqs.py`

---

## 0. Cross-cutting decisions

### 0.1 Candidate lookup path

Every revisit candidate is identified by a label such as `"rank01_frame_0432_d1"`.
The parsing + lookup pipeline is:

```
label  ──► regex match: r"rank(\d+)_frame(\d+)_d(\d+)"
            │
            ▼
         rank=1, frame_n=432, d_id=1
            │
            ├── seg JSON: chip_N/seg/frame_0432.json
            │    └── detections[d_id]          # d_id IS the Python 0-based index
            │         verified: det[1] → stage dist ≈ 86 µm from revisit coords
            │                   det[0] → stage dist ≈ 548 µm  (wrong)
            │
            └── raw frame: chip_N/scan_10x/frame_0432.jpg
```

**d_id is a 0-based direct array index into `detections[]`.**
Verified empirically for ranks 01–05 and rank 04 (d28): `detections[d_id]` gives the
centroid closest to the stage position in `revisit_50x.json["points"]` (residuals
86–143 µm, consistent with parcentric shift at 50x).

Error handling (in `io_utils.parse_revisit_label`):

| Failure | Behaviour |
|---|---|
| Label doesn't match regex | Raise `ValueError` with full label in message |
| `seg/frame_NNNN.json` missing | Raise `FileNotFoundError` with path |
| `d_id >= len(detections)` | Raise `IndexError`: "d_id={d_id} but frame has {n} detections" |
| `detections[d_id]` missing `contour` | Raise `KeyError` with field name and label |
| Contour has < 3 points | Raise `ValueError`: "contour too small to form a mask" |

No silent skips anywhere. All errors propagate with the label in context.

### 0.2 Classification filtering policy

Two separate policies — one per metric — documented here and as module-level
docstrings in `lhrr.py` and `eop.py`.

**LHRR — filter to target material:**
LHRR is the "is this flake worth stacking?" signal. Computing it on a
`classification = "non-hBN"` candidate in an hBN run is misleading.
Default: check `detection["classification"]` against a per-material allowlist.
If the classification is not in the allowlist, set all LHRR output fields to
`null` and set `lhrr_skip_reason = "classification_mismatch"`.

Default allowlists (configurable constant `LHRR_CLASSIFICATION_ALLOWLIST` in
`lhrr.py`):
```python
LHRR_CLASSIFICATION_ALLOWLIST: dict[str, list[str]] = {
    "hbn_medium":             None,   # None = accept all (lab to tighten once values are known)
    "graphene_thin_90nm":     None,
}
```
`None` means no filtering (accept any classification string). The lab should
update these once they have enumerated the classifier's full output vocabulary.
Note: rank01 in chip_0 hBN is classified `"non-hBN"` but was still revisited
at 50x (the pipeline's tier system uses different criteria). Setting `None`
by default avoids silently discarding candidates.

**EOP occupancy — include ALL detections regardless of classification:**
A `"non-hBN"` flake still physically obstructs the stamp. The occupancy map
must include every detection from every seg frame, irrespective of
`classification`. This is documented at the top of `eop.py`:
```
# EOP occupancy policy: ALL detections from all seg frames are included in the
# chip occupancy mask, regardless of `classification`. Non-target material
# physically obstructs stamp pickup just as much as target material.
```

### 0.3 Multi-run handling

**One run at a time.** `flake_metrics.py` accepts a single run directory as
`argv[1]`. The user runs it once per archive.

Material is auto-detected per-chip from `chip_N/seg/summary.json →
params.material`. Verified consistent across chips within a run
(`hbn_medium` for all hBN chips, `graphene_thin_90nm` for all graphene chips).
If a run has mixed materials (future-proof), each chip uses its own material
string independently.

**Obstruction weight estimation** (for EOP) uses one formula across both materials:
```
contrast_delta = |mean_green_flake_corrected - substrate_baseline_green|
// raw delta accumulated per detection into obstruction_map
// after all detections in the chip are processed:
obstruction_weight = contrast_delta / max(contrast_delta_across_chip)   // normalised to [0, 1]
```
This is material-agnostic: hBN appears brighter than substrate (positive delta),
graphene appears darker (negative delta, abs handles it). The per-chip
normalisation keeps weights in [0, 1] regardless of illumination conditions.
No material-specific calibration needed. (LHRR does not use obstruction weight.)

---

## 1. LHRR — Largest Homogeneous Rectangular Region

### 1.1 What it fixes

Global entropy + gradient energy penalises a flake with a dirty edge even if
its interior is pristine. LHRR finds the largest axis-aligned, defect-free
sub-rectangle and reports that area and fraction, so the lab can rank by
*usable* area rather than whole-flake score.

### 1.2 Pseudocode

```
FUNCTION compute_lhrr(chip_dir, label, flatfield, pixel_um, params):

  // Step 0: classification gate
  det = lookup_detection(chip_dir, label)
  if classification_allowlist is not None:
      if det.classification not in allowlist:
          return null_result("classification_mismatch")

  // Step 1: load and correct raw frame
  frame_n, d_id = parse_label(label)
  raw_jpg = chip_dir/scan_10x/frame_{frame_n:04d}.jpg
  raw = cv2.imread(raw_jpg)                             // BGR uint8
  corrected = clip(raw * flatfield, 0, 255).astype(uint8)

  // Step 2: build flake binary mask from contour
  contour = det.contour                                 // [[x,y], ...] pixel coords
  mask = zeros(H, W, bool)
  cv2.fillPoly(mask, [np.int32(contour)], 1)
  // Dilate slightly for border buffer (kernel = 5px) so border pixels
  // don't contaminate variance statistics
  mask_dilated = cv2.dilate(mask, kernel=5px_disk)

  // Step 3: crop to bounding box + padding
  x, y, w, h = det.bbox
  pad = max(CROP_PAD_PX, kernel_px + 2)
  roi = slice(max(0, y-pad) : min(H, y+h+pad),
              max(0, x-pad) : min(W, x+w+pad))
  green_crop = corrected[roi, channel=GREEN]
  mask_crop  = mask[roi]
  mask_d_crop = mask_dilated[roi]

  // Step 4: local variance map on green channel
  //   variance = E[X^2] - E[X]^2 computed via box filter (fast, O(N))
  kernel_px = odd_round(VARIANCE_KERNEL_UM / pixel_um)   // default 5 µm -> 7 px
  green_f32 = green_crop.astype(float32)
  mean_map  = cv2.filter2D(green_f32, kernel=box(kernel_px)) / kernel_px^2
  mean_sq   = cv2.filter2D(green_f32^2, kernel=box(kernel_px)) / kernel_px^2
  var_map   = clip(mean_sq - mean_map^2, 0, inf)        // clamp numerical noise

  // Step 5: adaptive threshold
  in_mask_var = var_map[mask_dilated_crop]
  if len(in_mask_var) == 0:
      return null_result("empty_mask")
  threshold = percentile(in_mask_var, VARIANCE_PERCENTILE)  // default 25th
  clean_mask = (var_map < threshold) AND mask_crop          // strict flake boundary

  // Step 6: largest axis-aligned rectangle (maximal histogram method)
  rect = max_rect_in_binary_mask(clean_mask)   // returns (x0, y0, w, h)
  if rect is None:
      return null_result("no_clean_region")

  // Step 7: convert coordinates back to frame space and stage space
  lhrr_bbox_frame = (rect.x + roi.x_offset, rect.y + roi.y_offset, rect.w, rect.h)
  lhrr_bbox_stage = pixel_bbox_to_stage(lhrr_bbox_frame, frame_meta, pixel_um)

  RETURN:
    lhrr_area_px     = rect.w * rect.h
    lhrr_area_um2    = lhrr_area_px * pixel_um^2
    lhrr_fraction    = lhrr_area_px / det.size_px
    lhrr_bbox_frame  = lhrr_bbox_frame
    lhrr_bbox_stage  = lhrr_bbox_stage
    variance_threshold_used = threshold
    kernel_size_px   = kernel_px
    lhrr_skip_reason = null
```

### 1.3 Maximal rectangle algorithm

Implemented from scratch in `src/lhrr.py`. Algorithm:

```
FUNCTION max_rect_in_binary_mask(mask):
  // Standard "largest rectangle in histogram" applied row-by-row.
  // O(rows * cols).

  h_map = zeros(cols, int)          // height of consecutive True pixels above
  best_area = 0
  best_rect = None

  FOR each row r in mask:
      FOR each col c:
          h_map[c] = h_map[c] + 1 if mask[r,c] else 0

      // Largest rectangle in h_map histogram (stack-based, O(cols)):
      rect = largest_rect_in_histogram(h_map, row=r)
      if rect.area > best_area:
          best_area = rect.area
          best_rect = rect

  RETURN best_rect

FUNCTION largest_rect_in_histogram(heights, row):
  // Classic stack algorithm. Returns (col_left, row_top, width, height).
  stack = []      // stores column indices
  best = (0, 0, 0, 0)
  FOR i in range(len(heights) + 1):
      h = heights[i] if i < len(heights) else 0
      while stack and heights[stack[-1]] > h:
          height = heights[stack.pop()]
          width  = i if not stack else i - stack[-1] - 1
          col_left = 0 if not stack else stack[-1] + 1
          if height * width > best.area:
              best = (col_left, row - height + 1, width, height)
      stack.append(i)
  RETURN best
```

Why this over alternatives:
- **Brute force O(N^2 * M^2)**: too slow for 1824×1216 crops.
- **Rotating calipers / convex hull**: finds largest convex rectangle, not
  axis-aligned. LHRR needs axis-aligned for stamp geometry.
- **Iterative erosion**: approximate and slow.
- **Histogram method O(R*C)**: exact, fast, well-understood.

### 1.4 Tunable parameters

| Constant | Default | Meaning |
|---|---|---|
| `VARIANCE_KERNEL_UM` | `5.0` | Spatial scale of surface-defect detection (µm). 5 µm ≈ 7 px. Use smaller to catch fine contamination; larger to ignore it. |
| `VARIANCE_PERCENTILE` | `50` | Percentile of in-mask variance used as clean/dirty threshold. Lower = stricter (smaller LHRR). Empirically validated at p25/p50/p75 on chip_0 hBN; p50 chosen — see §1.7. |
| `CROP_PAD_PX` | `30` | Padding around detection bbox for variance computation context. |
| `LHRR_CLASSIFICATION_ALLOWLIST` | `{material: None}` | Per-material classification strings to accept. `None` = accept all. |

### 1.5 Failure modes and detection

| Failure mode | Detection | Result |
|---|---|---|
| Entire flake is dirty (low-contrast substrate) | `lhrr_fraction < 0.05` | Valid result; low fraction signals problem |
| Flake smaller than kernel | `det.size_px < kernel_px^2` | Log warning; reduce kernel to 3×3 and retry |
| Contour self-intersects | `cv2.fillPoly` handles this | No special action needed |
| All variance below threshold | All pixels clean; LHRR = full flake bbox ∩ mask | Valid result |
| Raw frame missing from disk | `FileNotFoundError` with path | Propagate, skip chip with error log |

### 1.6 Validation example

**Flake selection from chip_0 hBN — size-based (queried from seg JSONs):**
- **Small:** `rank04_frame_0389_d28` — 505 µm² (actual smallest candidate in chip_0)
- **Median:** `rank13_frame_0149_d5` — 978 µm² (actual median by area, index 10 of 21)
- **Large:** `rank02_frame_0257_d2` — 4556 µm² (actual largest candidate in chip_0)

For each, generate `outputs/lhrr/chip_0_{label}.png` with 4 panels before
scaling to all chips.

**Pass/fail criteria for the 4-panel figure:**
1. Panel 1 (corrected + mask): mask outline should align with visible flake boundary
2. Panel 2 (variance heatmap): bright spots should correspond to visually dirty regions
3. Panel 3 (clean mask): should be a non-trivial region inside the flake, not empty, not the whole flake
4. Panel 4 (LHRR bbox): bbox must be fully inside the clean mask and clearly axis-aligned

If any panel fails visually, the metric is wrong — do not proceed to EOP.

### 1.7 Percentile calibration (empirical, chip_0 hBN)

A sensitivity sweep at p25 / p50 / p75 was run on the three validation flakes.
Results (LHRR area, strict-interior threshold sampling — see §1.7a):

| Flake | p25 area | p25 frac | p50 area | p50 frac | p75 area | p75 frac |
|---|---|---|---|---|---|---|
| small  (505 µm²)  |  33 µm² | 0.065 | 102 µm² | 0.201 | 216 µm² | 0.428 |
| median (978 µm²)  |  74 µm² | 0.076 | 109 µm² | 0.112 | 197 µm² | 0.202 |
| large  (4556 µm²) | 166 µm² | 0.037 | 469 µm² | 0.103 | 1382 µm² | 0.303 |

**p25** — Clean mask is fragmented (especially on the large flake with its noisy
interior patches). LHRR undershoots the visually usable area; fractions are
~4–8%.

**p50** — Clean mask is contiguous on all three flakes. LHRR rectangles land in
the visually flat interior; fractions are ~10–20%, consistent with real hBN
flakes whose edges are inherently rough. Chosen as default.

**p75** — Threshold becomes permissive enough to accept genuine edge variance.
On the large flake the LHRR box extends into the ragged border (frac=0.30).

**Decision:** `VARIANCE_PERCENTILE = 50`. Lab should re-run the sweep if a new
material preset or illumination condition is introduced.

#### 1.7a Threshold sampling fix

Variance map is computed over the dilated-mask crop (so border pixels receive
full box-filter context), but the percentile threshold is sampled from
`var_map[mask_crop]` — strict interior only. Buffer pixels straddle the flake
edge and have structurally elevated variance (~2.8× interior mean on the small
flake); including them in the sample inflates or deflates the threshold depending
on whether high-variance outliers pull the tail. Effect is flake-dependent:
+6.3% on the small flake (modest), −6.9% on the large flake (meaningful,
reduces clean-pixel count by 211). Fix is correct in principle and material on
high-contrast-edge flakes.

---

## 2. EOP — Ease of Pickup

### 2.1 What it fixes

A beautiful flake surrounded by tall neighbours is physically unusable — the
stamp clips the neighbours during pickup. The current pipeline has no spatial
awareness of neighbours. EOP surfaces this risk.

### 2.2 Pseudocode

```
FUNCTION build_chip_occupancy(chip_dir, flatfield, pixel_um, params):

  // Load scan bounds from scan_meta.json
  meta = load_scan_meta(chip_dir/scan_10x/scan_meta.json)
  x_min, x_max = meta.x_min_um, meta.x_max_um
  y_min, y_max = meta.y_min_um, meta.y_max_um
  stage_px = OCCUPANCY_STAGE_PX_UM   // default 5.0 µm/px (see §2.5 memory budget)

  // Allocate maps
  cols = ceil((x_max - x_min) / stage_px) + 1
  rows = ceil((y_max - y_min) / stage_px) + 1
  occ_map        = zeros(rows, cols, uint8)     // 255 = occupied, 0 = empty
  obstruction_map = zeros(rows, cols, float32)  // raw contrast_delta, normalised later

  // Substrate baseline: sample background from detection-masked frames (see §2.4)
  substrate_baseline = estimate_substrate_baseline(chip_dir, flatfield)

  // Accumulate ALL detections from all seg frames
  frames_by_n = {f["n"]: f for f in meta.frames}

  FOR each seg/frame_NNNN.json in chip_dir/seg/:
      frame_n = parse frame index from filename
      fr = frames_by_n[frame_n]
      raw = cv2.imread(chip_dir/scan_10x/frame_{frame_n:04d}.jpg)
      corrected = apply_flatfield(raw, flatfield)
      green = corrected[:,:,GREEN]

      FOR each detection det in frame_json.detections:
          // Project detection center to stage coords
          cx, cy = det.center
          sx = fr.x_um + (cx - W/2) * pixel_um
          sy = fr.y_um + (cy - H/2) * pixel_um
          // Project full contour to stage coords, then to map pixels
          stage_contour = project_contour_to_stage(det.contour, fr, pixel_um)
          map_contour   = ((stage_contour - [x_min, y_min]) / stage_px).astype(int)
          // Clip to map bounds
          map_contour = clip(map_contour, [0,0], [cols-1, rows-1])
          cv2.fillPoly(occ_map, [map_contour], 255)
          // Obstruction weight: mean green in corrected flake mask
          local_mask = zeros(H, W, bool)
          cv2.fillPoly(local_mask, [int32(det.contour)], 1)
          mean_green_flake = mean(green[local_mask])
          contrast_delta = abs(mean_green_flake - substrate_baseline)
          // Write raw delta into obstruction_map (max-accumulate: strongest contrast wins)
          map_mask = zeros(rows, cols, bool)
          cv2.fillPoly(map_mask, [map_contour], 1)
          obstruction_map = where(map_mask, maximum(obstruction_map, contrast_delta), obstruction_map)

  // Normalise obstruction_map to [0, 1] by chip-wide maximum
  max_contrast = max(obstruction_map)
  if max_contrast > 0:
      obstruction_map /= max_contrast

  RETURN occ_map, obstruction_map, (x_min, y_min, stage_px)

FUNCTION compute_eop(candidate_label, chip_dir, occ_map, obstruction_map,
                     map_origin, stage_px, pixel_um):

  det = lookup_detection(chip_dir, candidate_label)
  fr  = frames_by_n[frame_n_for_label]
  stage_contour = project_contour_to_stage(det.contour, fr, pixel_um)
  stage_center  = [fr.x_um + (det.center[0] - W/2) * pixel_um,
                   fr.y_um + (det.center[1] - H/2) * pixel_um]
  stage_cx, stage_cy = stage_center

  // Build a "this candidate" mask on the occupancy map
  cand_map_contour = ((stage_contour - map_origin) / stage_px).astype(int)
  cand_mask = zeros_like(occ_map, bool)
  cv2.fillPoly(cand_mask, [cand_map_contour], 1)

  // Clearance: distance from candidate boundary to nearest OTHER detection
  // Remove candidate from occupancy, run distance transform on free space.
  occ_without_cand = occ_map.copy()
  occ_without_cand[cand_mask] = 0
  // ~occ_without_cand: uint8 bitwise NOT. occ_map is {0, 255}, so ~255 = 0 and
  // ~0 = 255 in uint8 arithmetic — free space (0) becomes 255 (foreground for
  // distanceTransform), occupied pixels (255) become 0 (barrier). Correct polarity.
  dist_map = cv2.distanceTransform(~occ_without_cand, cv2.DIST_L2, 5)
  // Sample dist_map at candidate boundary pixels
  boundary = cand_mask XOR cv2.erode(cand_mask, 3x3)
  clearance_px = min(dist_map[boundary]) if any(boundary) else 0
  clearance_um = clearance_px * stage_px

  // Weighted obstruction: sum over non-candidate pixels in obstruction_map
  // within MAX_OBSTRUCTION_RADIUS_UM of candidate centroid.
  // Iterate only pixels where:
  //   (obstruction_map > 0)        — non-empty
  //   AND (cand_mask == 0)         — exclude the candidate itself
  //   AND (d_j_um <= MAX_OBSTRUCTION_RADIUS_UM)  — bounded radius
  // For each such pixel j at stage distance d_j_um from stage_center:
  //   weighted_obstruction += obstruction_map[j] / max(d_j_um, 1)^2
  // Implemented via a bounding-box crop around the candidate centroid to
  // avoid iterating the full map (O(radius/stage_px)^2 instead of O(rows*cols)).
  weighted_obstruction = sum_weighted_obstruction(
      stage_cx, stage_cy, cand_mask,
      obstruction_map, map_origin, stage_px,
      radius_um=MAX_OBSTRUCTION_RADIUS_UM)

  // EOP score
  eop_score = clearance_um / (1 + weighted_obstruction)
  // High = easy to pick up (large clearance, thin/few neighbours)
  // Low  = hard (close/thick neighbours)

  RETURN clearance_um, weighted_obstruction, eop_score
```

### 2.3 EOP score formula and justification

```
eop_score = clearance_um / (1 + weighted_obstruction)
```

- **Numerator** `clearance_um`: the gap between the candidate's physical edge
  and the nearest obstruction. A stamp typically needs ≥50 µm clearance to
  avoid clipping; scores below 50 are physically marginal.
- **Denominator** `1 + weighted_obstruction`: penalises by cumulative thickness
  of neighbours weighted by their closeness. The `+1` prevents division-by-zero
  and ensures that a flake with zero obstructions scores exactly `clearance_um`
  (interpretable: "this many microns of safe working space").
- **High score** = far from obstructions and/or obstructions are thin → easy pickup.
- **Low score** = crowded or thick neighbours → risky pickup.

Documented as a module-level docstring at the top of `src/eop.py`.

### 2.4 Substrate baseline estimation

```
FUNCTION estimate_substrate_baseline(chip_dir, flatfield):
  // Sample up to BASELINE_SAMPLE_FRAMES (default 20) frames.
  // Prefer frames from the first and last 10% of the scan index range
  // (chip edges are typically sparser — lower flake density).
  //
  // For each sampled frame:
  //   1. corrected = apply_flatfield(cv2.imread(frame_jpg), flatfield)
  //   2. combined_mask = zeros(H, W, bool)
  //      for each detection in seg/frame_NNNN.json (if file exists):
  //          cv2.fillPoly(combined_mask, [int32(det.contour)], 1)
  //   3. free_pixels = corrected[:, :, GREEN][~combined_mask]
  //      if len(free_pixels) < MIN_BASELINE_PIXELS (default 1000):
  //          skip this frame
  //   4. baseline_estimate_i = mean(free_pixels)
  //
  // substrate_baseline = median(baseline_estimate_i across accepted frames)
  //
  // Fallback: if fewer than 3 frames are accepted, use the
  // 5th percentile of per-frame full-frame mean-green values
  // (biased on dense chips but avoids a zero baseline).
  // Log a warning if fallback is used.
```

Masking detections before computing the mean removes flake pixels from the
substrate estimate, which matters on dense chips where the naive frame mean
would be pulled toward flake brightness.

### 2.5 Memory budget

Actual chip scan extents measured from `scan_meta.json` across all 8 chips:

| Run | Chip | X range | Y range | At 5 µm/px | Uint8 MB |
|---|---|---|---|---|---|
| hBN | chip_0 | 33.8 mm | 15.4 mm | 6760×3080 | 20.8 |
| hBN | chip_1 | 38.9 mm | 18.5 mm | 7780×3700 | 28.8 |
| hBN | chip_2 | 28.6 mm | 18.5 mm | 5720×3700 | 21.2 |
| hBN | chip_3 | 36.3 mm | 17.0 mm | 7260×3400 | 24.7 |
| hBN | chip_4 | 33.9 mm | 13.1 mm | 6780×2620 | 17.8 |
| hBN | chip_5 | 35.6 mm | 17.0 mm | 7120×3400 | 24.2 |
| Gr  | chip_0 | 26.0 mm | 13.9 mm | 5200×2780 | 14.5 |
| Gr  | chip_1 | 25.4 mm | 13.9 mm | 5080×2780 | 14.1 |

**At 5 µm/stage-px**: occupancy map (uint8) = 14–29 MB per chip. Obstruction map
(float32) = 56–116 MB per chip. Both are processed one chip at a time and
discarded after EOP scores are written.

**Why NOT 2 µm/stage-px (as given in the spec as an example):**
At 2 µm/px the occupancy map alone is 90–180 MB per chip; obstruction map
(float32) would be 360–720 MB. For 6 chips simultaneously that's multi-GB.
5 µm/px is well above the nyquist for stamp geometry (~100 µm stamp, ~50 µm
target clearance threshold) and is safely within budget on a typical workstation.
The stage_px resolution is exposed as `OCCUPANCY_STAGE_PX_UM` (default `5.0`)
so the lab can tighten it if they have the RAM.

### 2.6 Tunable parameters

| Constant | Default | Meaning |
|---|---|---|
| `OCCUPANCY_STAGE_PX_UM` | `5.0` | Occupancy map resolution (µm/px). |
| `MAX_OBSTRUCTION_RADIUS_UM` | `500.0` | Radius for weighted_obstruction sum. |
| `BASELINE_SAMPLE_FRAMES` | `20` | Max frames for substrate baseline estimate. |
| `EOP_MIN_CLEARANCE_UM` | `50.0` | Below this, flag `clearance_warning = True`. |

### 2.7 Failure modes

| Failure mode | Detection | Result |
|---|---|---|
| Candidate contour projects outside map bounds | Clip and log warning | Reduced accuracy but continues |
| All frames have detections (no clean baseline) | Fall back to 5th percentile of frame means | Log fallback |
| Candidate has no boundary pixels (1px flake) | `clearance_um = 0` | Valid; will rank last |
| dist_map all-zero (entire map occupied) | `clearance_um = 0` | Valid; extreme case |

### 2.8 Validation example

Same three flakes as LHRR validation (chip_0, size-based selection):
- **Small:** `rank04_frame_0389_d28` (505 µm²)
- **Median:** `rank13_frame_0149_d5` (978 µm²)
- **Large:** `rank02_frame_0257_d2` (4556 µm²)

For each, save `outputs/eop/chip_0_{label}.png` with:
- Chip occupancy map with candidate in green, all other detections in red
- Text annotation: `clearance={X:.0f} um  obstruction={Y:.2f}  eop={Z:.1f}`

Expected outcome: the three candidates should show meaningfully different
surroundings in the occupancy map, and EOP scores should rank them accordingly.

---

## 3. Integration — `flake_metrics.py`

### 3.1 Processing flow

```
argv[1] = run_dir

load flatfield (from FLATFIELD_PATH constant or auto-locate)

FOR each chip_N in run_dir (sorted):
    print "Processing chip_N ..."
    material = chip_N/seg/summary.json -> params.material
    candidates = chip_N/seg/revisit_50x.json -> points
    n_candidates = len(candidates)

    // EOP Phase 1: build occupancy + obstruction maps (reads all seg frames + JPEGs)
    // NOTE: this step takes ~10-15 min per run; prints per-frame progress to stdout.
    occ_map, obstruction_map, map_meta = build_chip_occupancy(chip_N, ...)

    // Save occupancy QC image
    save outputs/eop/chip_N_occupancy.png

    results = []
    FOR each candidate in candidates:
        label = candidate["label"]
        // LHRR
        lhrr = compute_lhrr(chip_N, label, flatfield, ...)
        // EOP
        eop  = compute_eop(label, chip_N, occ_map, obstruction_map, ...)
        results.append({**lhrr, **eop, "label": label, "material": material,
                        "stage_x": candidate["x"], "stage_y": candidate["y"]})
        print f"  {label}: lhrr={lhrr_area_um2:.0f} um2 (frac={lhrr_fraction:.2f}), eop={eop_score:.1f}"

    // Write per-chip JSON (augments, does not overwrite revisit_50x.json)
    write outputs/metrics/chip_N_metrics.json

// Aggregate CSV across all chips, sorted by composite score.
// Normalise after the chip loop so both signals use run-wide maxima.
lhrr_area_norm      = lhrr_area_um2 / max(lhrr_area_um2 across all candidates in run)
eop_score_normalised = eop_score    / max(eop_score    across all candidates in run)
composite = COMPOSITE_W_LHRR * lhrr_area_norm
          + COMPOSITE_W_EOP  * eop_score_normalised
write outputs/ranked_candidates.csv   // sorted descending by composite
```

### 3.2 Composite score formula

```
composite = COMPOSITE_W_LHRR * lhrr_area_norm
          + COMPOSITE_W_EOP  * eop_score_normalised
```

Where:
- `lhrr_area_norm = lhrr_area_um2 / max(lhrr_area_um2 across all candidates in run)`
- `eop_score_normalised = eop_score / max(eop_score across all candidates in run)`

Both normalised after processing all chips, using run-wide maxima.
`lhrr_fraction` (area / total flake area) is retained in the per-candidate JSON
for diagnostic use but is not the ranking signal — fraction conflates flake size
with cleanliness and would bias ranking toward smaller, uniformly-dirty flakes.
`lhrr_area_norm` ranks by absolute usable area, which is what the lab cares about.

Default weights: `COMPOSITE_W_LHRR = 0.5`, `COMPOSITE_W_EOP = 0.5`.
Documented at the top of `flake_metrics.py`. Both weights are constants.

**Note:** composite scores are run-relative — the best candidate in any run
scores near 1.0 by construction (both normalised terms max at 1.0). Scores
are not directly comparable across runs or material presets.

---

## 4. Module structure

```
src/
  io_utils.py
    parse_revisit_label(label)         -> (rank, frame_n, d_id)
    lookup_detection(chip_dir, label)  -> (det_dict, frame_meta)
    apply_flatfield(raw, ff)           -> corrected uint8
    build_flake_mask(det, H, W)        -> bool ndarray
    project_contour_to_stage(...)      -> float32 ndarray
    pixel_bbox_to_stage(...)           -> (x, y, w, h) in µm
    load_scan_meta(path)               -> dict (cached)

  lhrr.py
    VARIANCE_KERNEL_UM = 5.0
    VARIANCE_PERCENTILE = 25
    CROP_PAD_PX = 30
    LHRR_CLASSIFICATION_ALLOWLIST = {...}
    compute_lhrr(chip_dir, label, flatfield, pixel_um, ...)  -> dict
    max_rect_in_binary_mask(mask)      -> (x, y, w, h) | None
    _largest_rect_in_histogram(...)    -> (col_left, row_top, w, h)

  eop.py
    # EOP occupancy policy: ALL detections from all seg frames are included in the
    # chip occupancy mask, regardless of `classification`. Non-target material
    # physically obstructs stamp pickup just as much as target material.
    # obstruction_map is a visibility-based proxy for obstruction risk — it uses
    # optical contrast magnitude as a proxy for physical height, normalised per chip.
    OCCUPANCY_STAGE_PX_UM = 5.0
    MAX_OBSTRUCTION_RADIUS_UM = 500.0
    BASELINE_SAMPLE_FRAMES = 20
    build_chip_occupancy(chip_dir, flatfield, pixel_um, ...) -> (occ_map, obstruction_map, map_meta)
    estimate_substrate_baseline(chip_dir, flatfield)          -> float
    compute_eop(label, chip_dir, occ_map, obstruction_map, map_meta, ...)  -> dict

  flake_metrics.py   (entrypoint)
    FLATFIELD_PATH = ...
    COMPOSITE_W_LHRR = 0.5
    COMPOSITE_W_EOP  = 0.5
    main(run_dir)
```

---

## 5. Specific test flakes for validation

All from `chip_0` of the hBN run (SF121 D-J, `run_20260505_1616`).
Chosen by actual `size_um2` queried from seg JSONs before implementation —
these are the ground truth for all diagnostic figures.

| Flake | Label | `size_um2` | Selection method |
|---|---|---|---|
| Small | `rank04_frame_0389_d28` | 505 µm² | Actual smallest candidate in chip_0 |
| Median | `rank13_frame_0149_d5` | 978 µm² | Actual median by area (index 10 of 21) |
| Large | `rank02_frame_0257_d2` | 4556 µm² | Actual largest candidate in chip_0 |

For LHRR: 4-panel figure per flake saved before any batch processing.
For EOP: chip-level occupancy + candidate highlight per flake saved before batch.
Both must be visually convincing before scaling up.
