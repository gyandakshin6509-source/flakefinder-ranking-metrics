# FlakeFinder Metrics, Phase 2 Implementation Plan

> This is the Phase 2 plan as it stood on 2026-05-08. Some of it changed once I
> started implementing. `implementation-notes.md` and `README.md` describe what
> the code actually does now.

**Status:** Awaiting review before any code gets written.
**Prereq scripts run:** `src/recon.py`, `src/_check_plan_prereqs.py`

---

## 0. Cross-cutting decisions

### 0.1 Candidate lookup path

Every revisit candidate is identified by a label like `"rank01_frame_0432_d1"`.
Parsing and lookup goes like this:

```
label  ──► regex match: r"rank(\d+)_frame(\d+)_d(\d+)"
            │
            ▼
         rank=1, frame_n=432, d_id=1
            │
            ├── seg JSON: chip_N/seg/frame_0432.json
            │    └── detections[d_id]          # d_id IS the Python 0-based index
            │         verified: det[1] -> stage dist ~86 um from revisit coords
            │                   det[0] -> stage dist ~548 um  (wrong)
            │
            └── raw frame: chip_N/scan_10x/frame_0432.jpg
```

**`d_id` is a plain 0-based index into `detections[]`.** Nothing in the data
documents this, so I checked it empirically on ranks 01 through 05 and on rank
04 (d28). In every case `detections[d_id]` gives the centroid closest to the
stage position recorded in `revisit_50x.json["points"]`, with residuals of 86 to
143 µm, which is about what you'd expect from parcentric shift at 50x. Picking
the wrong index puts you hundreds of microns off, so this is easy to tell apart.

Error handling lives in `io_utils.parse_revisit_label`:

| Failure | Behaviour |
|---|---|
| Label doesn't match regex | Raise `ValueError` with the full label in the message |
| `seg/frame_NNNN.json` missing | Raise `FileNotFoundError` with the path |
| `d_id >= len(detections)` | Raise `IndexError`: "d_id={d_id} but frame has {n} detections" |
| `detections[d_id]` missing `contour` | Raise `KeyError` with the field name and label |
| Contour has fewer than 3 points | Raise `ValueError`: "contour too small to form a mask" |

Nothing gets skipped silently anywhere. Every error carries the label with it.

### 0.2 Classification filtering policy

The two metrics need different policies here, so both are written up below and
also as module docstrings in `lhrr.py` and `eop.py`.

**LHRR filters to the target material.** LHRR is the "is this flake worth
stacking" signal, so computing it on a candidate whose `classification` is
`"non-hBN"` inside an hBN run doesn't mean much. The default behaviour checks
`detection["classification"]` against a per-material allowlist, and if it isn't
in the list, every LHRR output field is set to `null` with
`lhrr_skip_reason = "classification_mismatch"`.

Default allowlists, configurable through `LHRR_CLASSIFICATION_ALLOWLIST` in
`lhrr.py`:

```python
LHRR_CLASSIFICATION_ALLOWLIST: dict[str, list[str]] = {
    "hbn_medium":             None,   # None = accept all
    "graphene_thin_90nm":     None,
}
```

`None` means no filtering at all, so any classification string gets through.
These should get tightened once someone has enumerated everything the
classifier can actually emit. Worth noting that rank01 in chip_0 of the hBN run
is classified `"non-hBN"` and was still revisited at 50x, because the
pipeline's tier system uses different criteria. Defaulting to `None` avoids
quietly throwing away candidates like that.

**EOP occupancy includes every detection regardless of classification.** A
`"non-hBN"` flake still physically blocks the stamp. So the occupancy map takes
every detection from every seg frame no matter what it's classified as. This is
documented at the top of `eop.py`:

```
# EOP occupancy policy: ALL detections from all seg frames are included in the
# chip occupancy mask, regardless of `classification`. Non-target material
# physically obstructs stamp pickup just as much as target material.
```

### 0.3 Multi-run handling

**One run at a time.** `flake_metrics.py` takes a single run directory as
`argv[1]` and gets run once per archive.

Material is detected per chip from `chip_N/seg/summary.json -> params.material`.
I checked and it's consistent across chips within a run (`hbn_medium` for all
the hBN chips, `graphene_thin_90nm` for all the graphene ones), but each chip
reads its own value independently in case a mixed run ever shows up.

**Obstruction weight estimation** for EOP uses one formula for both materials:

```
contrast_delta = |mean_green_flake_corrected - substrate_baseline_green|
// raw delta accumulated per detection into obstruction_map
// after all detections in the chip are processed:
obstruction_weight = contrast_delta / max(contrast_delta_across_chip)   // -> [0, 1]
```

This works either way round, since hBN reads brighter than the substrate
(positive delta) and graphene reads darker (negative delta, and the absolute
value handles it). Normalising per chip keeps the weights in [0, 1] whatever
the illumination was like, so there's no material-specific calibration needed.
LHRR doesn't use obstruction weight at all.

---

## 1. LHRR, Largest Homogeneous Rectangular Region

### 1.1 What it fixes

Global entropy plus gradient energy penalises a flake with a dirty edge even
when its interior is pristine. LHRR finds the largest axis-aligned defect-free
sub-rectangle instead and reports its area and fraction, so the lab can rank by
*usable* area rather than by a whole-flake score.

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
  // Dilate slightly (kernel = 5px) so border pixels don't contaminate
  // the variance statistics
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
  //   variance = E[X^2] - E[X]^2 via box filter (fast, O(N))
  kernel_px = odd_round(VARIANCE_KERNEL_UM / pixel_um)   // default 5 um -> 7 px
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

Written from scratch in `src/lhrr.py`:

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

Why this and not something else:

- **Brute force** at O(N² M²) is far too slow for 1824x1216 crops.
- **Rotating calipers / convex hull** gives the largest convex rectangle, which
  isn't necessarily axis-aligned. LHRR needs axis-aligned because that's the
  stamp geometry.
- **Iterative erosion** is both approximate and slow.
- **Histogram method** at O(R*C) is exact, fast, and well understood.

### 1.4 Tunable parameters

| Constant | Default | Meaning |
|---|---|---|
| `VARIANCE_KERNEL_UM` | `5.0` | Spatial scale of surface-defect detection in µm. 5 µm is about 7 px. Smaller catches finer contamination, larger ignores it. |
| `VARIANCE_PERCENTILE` | `50` | Percentile of in-mask variance used as the clean/dirty threshold. Lower is stricter, so smaller LHRR. Tested at p25/p50/p75 on chip_0 hBN, see §1.7. |
| `CROP_PAD_PX` | `30` | Padding around the detection bbox so variance has context. |
| `LHRR_CLASSIFICATION_ALLOWLIST` | `{material: None}` | Per-material classification strings to accept. `None` accepts all. |

### 1.5 Failure modes and detection

| Failure mode | Detection | Result |
|---|---|---|
| Whole flake is dirty (low-contrast substrate) | `lhrr_fraction < 0.05` | Valid result, low fraction is the signal |
| Flake smaller than the kernel | `det.size_px < kernel_px^2` | Log a warning, drop to a 3x3 kernel and retry |
| Contour self-intersects | `cv2.fillPoly` handles it | Nothing special needed |
| All variance below threshold | Everything clean, LHRR = full bbox ∩ mask | Valid result |
| Raw frame missing from disk | `FileNotFoundError` with the path | Propagate, skip the chip with an error log |

### 1.6 Validation example

Three flakes from chip_0 of the hBN run, picked by actual `size_um2` queried
out of the seg JSONs before writing any metric code:

- **Small:** `rank04_frame_0389_d28`, 505 µm², the smallest candidate in chip_0
- **Median:** `rank13_frame_0149_d5`, 978 µm², the median by area (index 10 of 21)
- **Large:** `rank02_frame_0257_d2`, 4556 µm², the largest candidate in chip_0

Each one gets a 4-panel figure at `outputs/lhrr/chip_0_{label}.png` before
anything gets scaled up to all the chips. What each panel has to show:

1. Panel 1 (corrected image plus mask): the mask outline should line up with the
   visible flake boundary
2. Panel 2 (variance heatmap): bright spots should correspond to regions that
   look dirty
3. Panel 3 (clean mask): a non-trivial region inside the flake, not empty and
   not the entire flake
4. Panel 4 (LHRR bbox): the box has to sit fully inside the clean mask and be
   obviously axis-aligned

If any panel doesn't hold up visually then the metric is wrong and EOP waits.

### 1.7 Percentile calibration (chip_0 hBN)

Sweep at p25 / p50 / p75 on the three validation flakes, using strict-interior
threshold sampling (see §1.7a):

| Flake | p25 area | p25 frac | p50 area | p50 frac | p75 area | p75 frac |
|---|---|---|---|---|---|---|
| small  (505 µm²)  |  33 µm² | 0.065 | 102 µm² | 0.201 | 216 µm² | 0.428 |
| median (978 µm²)  |  74 µm² | 0.076 | 109 µm² | 0.112 | 197 µm² | 0.202 |
| large  (4556 µm²) | 166 µm² | 0.037 | 469 µm² | 0.103 | 1382 µm² | 0.303 |

**p25** fragments the clean mask, worst on the large flake with its noisy
interior patches. LHRR comes out under the visually usable area, fractions
around 4 to 8%.

**p50** keeps the clean mask contiguous on all three. The rectangles land in the
flat interior and fractions sit around 10 to 20%, which is about right for real
hBN flakes given how rough their edges naturally are. This is the default.

**p75** gets permissive enough to accept genuine edge variance. On the large
flake the box extends into the ragged border at frac 0.30.

**Decision:** `VARIANCE_PERCENTILE = 50`. Re-run the sweep if a new material
preset or illumination condition comes in.

#### 1.7a Threshold sampling fix

The variance map is computed over the dilated-mask crop so border pixels get
full box-filter context, but the percentile threshold is sampled from
`var_map[mask_crop]`, the strict interior only. Buffer pixels straddle the flake
edge and have structurally higher variance, roughly 2.8x the interior mean on
the small flake. Including them pulls the threshold around depending on whether
high-variance outliers drag the tail, and the effect varies by flake: +6.3% on
the small one, which is modest, and -6.9% on the large one, which costs 211
clean pixels and is not modest. The fix is right in principle and matters in
practice on flakes with high-contrast edges.

---

## 2. EOP, Ease of Pickup

### 2.1 What it fixes

A beautiful flake surrounded by tall neighbours is physically unusable, since
the stamp clips the neighbours on the way down. The current pipeline has no
spatial awareness of neighbours at all, so EOP is what surfaces that risk.

### 2.2 Pseudocode

```
FUNCTION build_chip_occupancy(chip_dir, flatfield, pixel_um, params):

  // Load scan bounds from scan_meta.json
  meta = load_scan_meta(chip_dir/scan_10x/scan_meta.json)
  x_min, x_max = meta.x_min_um, meta.x_max_um
  y_min, y_max = meta.y_min_um, meta.y_max_um
  stage_px = OCCUPANCY_STAGE_PX_UM   // default 5.0 um/px, see §2.5

  // Allocate maps
  cols = ceil((x_max - x_min) / stage_px) + 1
  rows = ceil((y_max - y_min) / stage_px) + 1
  occ_map        = zeros(rows, cols, uint8)     // 255 = occupied, 0 = empty
  obstruction_map = zeros(rows, cols, float32)  // raw contrast_delta, normalised later

  // Substrate baseline: sample background from detection-masked frames, §2.4
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
          // Write raw delta into obstruction_map (max-accumulate: strongest wins)
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

  // Clearance: distance from candidate boundary to nearest OTHER detection.
  // Remove candidate from occupancy, run distance transform on free space.
  occ_without_cand = occ_map.copy()
  occ_without_cand[cand_mask] = 0
  // ~occ_without_cand: uint8 bitwise NOT. occ_map is {0, 255}, so ~255 = 0 and
  // ~0 = 255 in uint8 arithmetic. Free space (0) becomes 255 (foreground for
  // distanceTransform), occupied pixels (255) become 0 (barrier). Correct polarity.
  dist_map = cv2.distanceTransform(~occ_without_cand, cv2.DIST_L2, 5)
  // Sample dist_map at candidate boundary pixels
  boundary = cand_mask XOR cv2.erode(cand_mask, 3x3)
  clearance_px = min(dist_map[boundary]) if any(boundary) else 0
  clearance_um = clearance_px * stage_px

  // Weighted obstruction: sum over non-candidate pixels in obstruction_map
  // within MAX_OBSTRUCTION_RADIUS_UM of candidate centroid.
  // Iterate only pixels where:
  //   (obstruction_map > 0)        non-empty
  //   AND (cand_mask == 0)         exclude the candidate itself
  //   AND (d_j_um <= MAX_OBSTRUCTION_RADIUS_UM)  bounded radius
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

### 2.3 EOP score formula and why

```
eop_score = clearance_um / (1 + weighted_obstruction)
```

The numerator, `clearance_um`, is the gap between the candidate's physical edge
and the nearest obstruction. A stamp generally wants at least 50 µm to avoid
clipping, so anything under 50 is physically marginal.

The denominator, `1 + weighted_obstruction`, penalises by how much neighbour
material there is, weighted by how close it is. The `+1` keeps it from dividing
by zero and also means a flake with no obstructions at all scores exactly
`clearance_um`, which reads nicely as "this many microns of safe working space".

So a high score means far from obstructions, or obstructions that are thin, or
both. A low score means crowded or thick neighbours. This goes in the
module-level docstring at the top of `src/eop.py`.

### 2.4 Substrate baseline estimation

```
FUNCTION estimate_substrate_baseline(chip_dir, flatfield):
  // Sample up to BASELINE_SAMPLE_FRAMES (default 20) frames.
  // Prefer frames from the first and last 10% of the scan index range,
  // since chip edges are usually sparser.
  //
  // For each sampled frame:
  //   1. corrected = apply_flatfield(cv2.imread(frame_jpg), flatfield)
  //   2. combined_mask = zeros(H, W, bool)
  //      for each detection in seg/frame_NNNN.json (if the file exists):
  //          cv2.fillPoly(combined_mask, [int32(det.contour)], 1)
  //   3. free_pixels = corrected[:, :, GREEN][~combined_mask]
  //      if len(free_pixels) < MIN_BASELINE_PIXELS (default 1000):
  //          skip this frame
  //   4. baseline_estimate_i = mean(free_pixels)
  //
  // substrate_baseline = median(baseline_estimate_i across accepted frames)
  //
  // Fallback: if fewer than 3 frames are accepted, use the 5th percentile of
  // per-frame full-frame mean-green values. Biased on dense chips but it
  // avoids a zero baseline. Log a warning if the fallback fires.
```

Masking out the detections before taking the mean is what keeps flake pixels
out of the substrate estimate. On a dense chip a naive frame mean gets pulled
toward flake brightness, which would quietly bias every contrast delta on that
chip.

### 2.5 Memory budget

Actual chip scan extents, measured from `scan_meta.json` across all 8 chips:

| Run | Chip | X range | Y range | At 5 µm/px | Uint8 MB |
|---|---|---|---|---|---|
| hBN | chip_0 | 33.8 mm | 15.4 mm | 6760x3080 | 20.8 |
| hBN | chip_1 | 38.9 mm | 18.5 mm | 7780x3700 | 28.8 |
| hBN | chip_2 | 28.6 mm | 18.5 mm | 5720x3700 | 21.2 |
| hBN | chip_3 | 36.3 mm | 17.0 mm | 7260x3400 | 24.7 |
| hBN | chip_4 | 33.9 mm | 13.1 mm | 6780x2620 | 17.8 |
| hBN | chip_5 | 35.6 mm | 17.0 mm | 7120x3400 | 24.2 |
| Gr  | chip_0 | 26.0 mm | 13.9 mm | 5200x2780 | 14.5 |
| Gr  | chip_1 | 25.4 mm | 13.9 mm | 5080x2780 | 14.1 |

At 5 µm per stage pixel the occupancy map (uint8) is 14 to 29 MB per chip and
the obstruction map (float32) is 56 to 116 MB. Both get processed one chip at a
time and thrown away once the EOP scores are written.

The spec gave 2 µm/px as an example and I'm not using it. At that resolution
the occupancy map alone is 90 to 180 MB per chip and the obstruction map would
be 360 to 720 MB, so six chips at once is multiple gigabytes. 5 µm/px is still
well above what the geometry needs, given a stamp around 100 µm and a clearance
threshold around 50 µm, and it fits comfortably on a normal workstation. It's
exposed as `OCCUPANCY_STAGE_PX_UM` so the lab can tighten it if they have the
RAM to spare.

### 2.6 Tunable parameters

| Constant | Default | Meaning |
|---|---|---|
| `OCCUPANCY_STAGE_PX_UM` | `5.0` | Occupancy map resolution in µm/px. |
| `MAX_OBSTRUCTION_RADIUS_UM` | `500.0` | Radius for the weighted_obstruction sum. |
| `BASELINE_SAMPLE_FRAMES` | `20` | Max frames for the substrate baseline estimate. |
| `EOP_MIN_CLEARANCE_UM` | `50.0` | Below this, flag `clearance_warning = True`. |

### 2.7 Failure modes

| Failure mode | Detection | Result |
|---|---|---|
| Candidate contour projects outside map bounds | Clip and log a warning | Less accurate but keeps going |
| Every frame has detections, no clean baseline | Fall back to 5th percentile of frame means | Log the fallback |
| Candidate has no boundary pixels (1px flake) | `clearance_um = 0` | Valid, it'll rank last |
| dist_map is all zero (whole map occupied) | `clearance_um = 0` | Valid, extreme case |

### 2.8 Validation example

Same three flakes as the LHRR validation:

- **Small:** `rank04_frame_0389_d28` (505 µm²)
- **Median:** `rank13_frame_0149_d5` (978 µm²)
- **Large:** `rank02_frame_0257_d2` (4556 µm²)

Each gets `outputs/eop/chip_0_{label}.png` with the chip occupancy map,
candidate in green and every other detection in red, plus a text annotation
reading `clearance={X:.0f} um  obstruction={Y:.2f}  eop={Z:.1f}`.

What I expect to see is three candidates with visibly different surroundings on
the occupancy map, and EOP scores that rank them the way the pictures suggest
they should be ranked.

---

## 3. Integration, `flake_metrics.py`

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
    // NOTE: takes ~10-15 min per run, prints per-frame progress to stdout.
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

where `lhrr_area_norm = lhrr_area_um2 / max(lhrr_area_um2 across the run)` and
`eop_score_normalised = eop_score / max(eop_score across the run)`. Both get
normalised after all the chips are done, using run-wide maxima.

`lhrr_fraction` (area over total flake area) stays in the per-candidate JSON as
a diagnostic but it isn't the ranking signal. Fraction mixes flake size
together with cleanliness, which would bias the ranking toward small flakes
that happen to be uniformly dirty. `lhrr_area_norm` ranks by absolute usable
area instead, which is the thing the lab actually cares about.

Default weights are `COMPOSITE_W_LHRR = 0.5` and `COMPOSITE_W_EOP = 0.5`, both
constants at the top of `flake_metrics.py`.

One consequence worth stating: composite scores are run-relative, so the best
candidate in any run scores near 1.0 by construction. They aren't directly
comparable across runs or material presets.

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
    pixel_bbox_to_stage(...)           -> (x, y, w, h) in um
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
    # obstruction_map is a visibility-based proxy for obstruction risk. It uses
    # optical contrast magnitude as a stand-in for physical height, normalised per chip.
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

## 5. Test flakes for validation

All from `chip_0` of the hBN run (SF121 D-J, `run_20260505_1616`), chosen by
actual `size_um2` queried from the seg JSONs before implementation started.
These are the ground truth for every diagnostic figure.

| Flake | Label | `size_um2` | How it was chosen |
|---|---|---|---|
| Small | `rank04_frame_0389_d28` | 505 µm² | Smallest candidate in chip_0 |
| Median | `rank13_frame_0149_d5` | 978 µm² | Median by area (index 10 of 21) |
| Large | `rank02_frame_0257_d2` | 4556 µm² | Largest candidate in chip_0 |

LHRR gets a 4-panel figure per flake before any batch processing. EOP gets a
chip-level occupancy plus candidate highlight per flake, also before batch.
Both have to look convincing before anything scales up.
