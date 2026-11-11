# Overhead visible-object counting

`OverheadObjectCounter` combines the existing `RGBDBackgroundSegmenter` with
appearance grouping for **near-solid, distinguishably colored deformable objects**.
It estimates how many visible color identities remain in an overhead current/empty
RGB-D pair, or groups a caller-supplied foreground mask with `detect_mask()`.
It does not recognize cloth, arms, or physical object identity.

Disconnected visible parts with matching color are grouped together. Touching
objects with distinct colors can be separated even without a depth step. Crop
edges are retained, and an observed unchanged scene returns zero candidates.
Fully hidden objects and objects outside the crop are outside the result's scope.

## Install and run

Detection requires NumPy plus the existing optional OpenCV dependency. Package
import itself remains NumPy-only. From an existing virtual environment:

```bash
python -m pip install -e '.[counting]'
python examples/example_object_count.py
python examples/example_object_count_mask.py
python -m unittest tests.test_object_count tests.test_object_count_mask -v
```

The self-contained example prints strict JSON for empty, overlapping/shaded/cropped,
imperfect-arm-mask, and thin-object-or-shadow scenes. It reads no private captures,
writes no files, makes no network requests, and issues no robot commands.

## In-memory API

```python
from depth_image_lens import OverheadObjectCounter

counter = OverheadObjectCounter(
    min_object_area=64,
    min_fragment_area=8,
    exclusion_guard_px=2,
    # Optional, scene-specific camera-Z separation limit. Not a universal height.
    max_reference_separation_m=0.15,
)
result = counter.detect(
    current_color,
    current_depth,
    background_color=empty_color,
    background_depth=empty_depth,
    color_order="RGB",                  # Or BGR, for BOTH color images.
    registered_to_depth=True,           # Caller declaration, not a registration operation.
    depth_scale_m=0.001,                # Both depth arrays: metres per stored count.
    arm_mask=arm_mask,                  # Optional bool HxW; True excludes.
    work_roi=work_roi,                  # Optional bool HxW; True searches.
    diagnostics=True,
)
print(result.count, result.candidate_count, result.state, result.reasons)
print(result.to_dict())                 # Summary only, no full-resolution arrays.
if result.candidate_count:
    visible_part = result.group_mask(1) # Fresh bool mask, potentially disconnected.
preview = result.render_debug(current_color, color_order="RGB")
```

`ObjectCountResult` and `OverheadObjectCounter` are exported from
`depth_image_lens`. Images/masks are never modified, noncontiguous views are
supported, and no resize, registration, depth filling, or color correction occurs
in the counter. The background segmenter can separately perform its explicitly
configured background compensation.

All four images must share the exact grid, fixed camera pose, crop, and scale.
Color must be uint8 HxWx3. Depth must be numeric optical-axis Z, not a rendered
color depth visualization. `depth_scale_m` is mandatory even for floating-point
metres (`1.0`). `16UC1` does not itself imply millimetres. Current/background depth
must use the same scale. Zero, negative, nonfinite, declared sentinel values, and
false validity masks mean missing measurements, never zero scene change.

Optional call-time bool masks are `arm_mask`, `exclusion_mask`, `work_roi`,
`color_valid_mask`, `background_color_valid_mask`, `depth_valid_mask`,
`background_depth_valid_mask`, and `background_check_mask`. All must match HxW.
The check mask declares pixels known to be unchanged background and inherits the
segmenter's reference-check requirements. `invalid_values` declares depth sentinels.
`diagnostics` is a strict bool.

## Supply an existing mask directly

Use `detect_mask()` when the foreground mask already exists. It **does not call
`RGBDBackgroundSegmenter.detect()`** and needs no empty-scene color image. The
existing `detect()` API is unchanged. Both entrypoints share grouping, fragment
filtering, arm guards, upper depth bands, stability checks, and result types.

```python
result = counter.detect_mask(
    current_color,
    foreground_mask,                    # Strict bool HxW, not uint8 0/255.
    color_order="RGB",
    depth=current_depth,                # Existing original sensor arrays.
    background_depth=empty_depth,
    depth_scale_m=0.001,
    registered_to_depth=True,
    observed_mask=mask_coverage,         # Optional: False means unknown, not empty.
    uncertain_mask=input_uncertainty,   # Optional: preserve upstream uncertainty.
    arm_mask=arm_mask,                   # Same exclusions and guard as detect().
    diagnostics=True,
)
```

The full signature is:

```python
counter.detect_mask(
    color, object_mask, *, color_order,
    depth=None, background_depth=None, depth_scale_m=None,
    registered_to_depth=False,
    arm_mask=None, exclusion_mask=None, work_roi=None,
    color_valid_mask=None, depth_valid_mask=None,
    background_depth_valid_mask=None, observed_mask=None, uncertain_mask=None,
    cleanup_changed_mask=None, invalid_values=(), diagnostics=False,
)
```

- `color` is uint8 HxWx3; all masks are bool HxW on the same registered grid,
  including crop. Numeric masks require explicit conversion by the caller. An
  instance-label map is not accepted as a binary mask.
- The current/empty depth pair is optional, but either **both or neither** must
  be supplied. When present, supply `depth_scale_m` and
  `registered_to_depth=True`. The same numeric, validity, sentinel, and missing
  depth rules as `detect()` apply. No depth image is synthesized from the mask.
- Signed `empty_Z - current_Z` provides support only above the configured
  `background_segmenter.parameters['depth_threshold_m']` and outside its
  `depth_margin_m`. Defaults remain 5 mm and 1 mm. The direct path reuses only
  these two settings, not background color thresholds, fusion, reference checks,
  compensation or segmentation area filters. Configure them through
  `OverheadObjectCounter(background_segmenter=RGBDBackgroundSegmenter(...))` if
  necessary; that instance is not run by `detect_mask()`.
- `observed_mask` defaults to all true and declares coverage of the external
  mask, including its false/background pixels. `color_valid_mask` and both depth
  validity masks further restrict actual observations. Missing pixels anywhere
  in the searchable domain still block an accepted count.
- `uncertain_mask` and `cleanup_changed_mask` default to all false. **Any** true
  pixel in the searchable domain blocks an accepted count, even outside the
  supplied foreground. Both remain in `result.uncertain_mask`; neither can
  contribute depth support. This deliberately conservative direct-input rule
  is stricter than the segmenter's built-in uncertainty checks, because a bare
  mask does not explain why its upstream decisions were uncertain.
- Known exclusions/ROI still define the visible searched domain. Uncertainty
  outside it does not block that domain's estimate, and excluded pixels are not
  reported as observed. Do not exclude real targets merely to obtain an integer.
- The mask's semantic selection and completeness are caller assumptions. A
  missing object, unwanted arm, merged colors or incorrect mask can still produce
  a wrong integer. There is no external-mask validation against empty RGB.

With only RGB and a binary mask, use:

```python
result = counter.detect_mask(current_color, foreground_mask, color_order="RGB")
# result.candidate_count and result.group_mask(...) still expose color hypotheses.
# result.count is None and state is insufficient: depth evidence was not provided.
```

Without a depth pair, scale/depth-validity/sentinel options are rejected rather
than silently ignored. An optional upper depth band cannot admit foreground with
unknown separation; in that case color hypotheses can also be withheld. A fully
observed empty mask with depth can return zero, conditional on external coverage
and completeness. A bare empty mask with no depth cannot certify zero.

### Count a cleaned mask without inventing evidence

```python
from depth_image_lens import cleanup_mask

# segmented is an already-computed RGBDBackgroundResult. Do not run it again.
protected = segmented.suppressed_mask | segmented.deferred_mask
repair = cleanup_mask(
    segmented.object_mask,
    max_hole_area=4,                     # Example only; validate at your resolution.
    observed_mask=segmented.outline_observed_mask,
    protected_mask=protected,
)
result = counter.detect_mask(
    current_color,
    repair.object_mask,
    color_order="RGB",
    depth=current_depth,
    background_depth=empty_depth,
    depth_scale_m=0.001,
    registered_to_depth=True,
    depth_valid_mask=segmented.depth_valid_mask,
    observed_mask=segmented.outline_observed_mask,
    uncertain_mask=segmented.uncertain_mask | protected,
    cleanup_changed_mask=repair.changed_mask,  # Includes removals, not just additions.
)
```

Preserve original color/depth validity, ROI and exclusions too when the upstream
pipeline used them. Keep the cleanup result and original mask for comparison.
Repaired geometry remains a hypothesis, even where raw depth is valid. A cleanup
that changes searched pixels therefore returns `count=None` while still exposing
`candidate_count`, masks and groups. Passing only the repaired mask and dropping
its uncertainty would conceal the inference. No cleanup is run implicitly.

Direct results report `parameters['input_mode'] == 'external_mask'`,
`background_segmenter=None`, explicit depth-evidence/settings provenance, and
`color_reference_status='not_used_external_mask'`. With diagnostics enabled,
`input_uncertain_mask` and `cleanup_changed_mask` show provenance within the
searched domain. `diagnostics['background']` is None; there is no fabricated
background-segmentation result. Original RGB-D calls report
`input_mode='rgbd_background'` and retain their segmentation summary.

## Read the result conservatively

| Field | Meaning |
|---|---|
| `count` | Conditional count estimate when the implemented evidence checks pass; `None` when unresolved. Not an exact physical count. |
| `candidate_count` | Number of retained visible color-group hypotheses, including ambiguous groups. Always an integer, not a physical lower or upper bound. |
| `state` | `estimate`, `no_visible_candidates`, `ambiguous`, or `insufficient`. |
| `groups` | Per-group area, visible fragment count/areas, RGB representative, normalized color, color spread, depth-support fraction, bounding box, crop flag, and reasons. |
| `label_map` | int32 HxW; 0 is unassigned, 1..candidate_count identify groups in this result. IDs are not stable temporal identities or RGB-ranked labels. |
| `object_mask` | Retained group pixels, exactly `label_map > 0`; hypotheses, including ambiguous groups. |
| `candidate_mask` | Supplied external-mask or background-segmenter candidates in the searchable, unexcluded domain before color/area filtering. |
| `rejected_mask` | Candidate pixels not retained in a group, including tiny fragments, dark colors, ambiguous assignments and budget overflow. |
| `uncertain_mask` | Segmentation uncertainty, missingness, unresolved assignments and rejected evidence. Can overlap object_mask and is not a probability map. |
| `excluded_mask` | Caller exclusions and optional high-separation pixels, expanded by the guard within the ROI. |
| `observed_mask` | Searchable pixels with usable color evidence and valid current/empty depth. Excluded pixels are not known empty. |
| `reasons` | Current blockers to accepting a count, such as missing depth, color ambiguity, rejected reference, or insufficient depth support. |
| `warnings` | Assumptions and permanent counterexamples that apply even when count is an integer. |
| `metrics` | Pixel accounting, nearby-threshold candidate counts, and color-reference status. |
| `diagnostics` | Optional depth separation, color features, support, dark/ambiguous/upper-band masks, and background summary. |

`to_dict()` produces a strict-JSON-compatible summary. No calibrated confidence
score or certified physical count interval is provided. `group_mask(id)` returns a
copy; unknown IDs raise ValueError. `render_debug` uses the shared mask renderer.
The label map is available for a caller's own per-group overlay.

A clean, fully observed empty scene returns `count=0`. An empty ROI, all-excluded
image, unusable reference or missing observations returns `count=None`, even if
`candidate_count=0`. A few valid excluded arm pixels do not themselves block the
visible-domain estimate; hidden objects behind them remain outside its scope.
Any missing color/depth elsewhere in the searchable domain blocks an accepted
count. This deliberately strict rule avoids treating a sensor hole as an empty
patch. Narrow the ROI or provide exclusions only for regions you genuinely intend
to omit, not to conceal uncertainty.

## Processing and tuning

1. Either supply an existing mask and its evidence with `detect_mask()`, or use
   `detect()` with `RGBDBackgroundSegmenter` to retain current/empty change evidence and its
   missingness/conflicts. The default is its union mode. Supply a configured
   `background_segmenter=RGBDBackgroundSegmenter(...)` to change background
   thresholds or fusion mode; its settings and provenance are included in output.
2. Remove caller arm/exclusion masks plus a square guard. Optionally also exclude
   pixels where `empty_Z - current_Z > max_reference_separation_m`. Unknown depth
   cannot pass that upper-band test and remains unresolved.
3. Form normalized encoded-RGB features `255 * RGB / sum(RGB)` on eligible
   candidates. This suppresses multiplicative brightness changes and handles red
   across a hue wrap without special hue arithmetic. Very dark pixels are not
   used for color identity.
4. Select deterministic bounded histogram modes and assign full-precision colors
   to nearby modes. There is no random subsampling, fixed forced object count,
   or transitive single-linkage merge through intermediate colors.
5. Retain connected fragments large enough to provide local evidence, then pool
   same-color fragments across the ROI. Check total support, every fragment's
   depth support, achromatic ambiguity, color spread, assignment margins, and
   count stability at nearby tolerances. Similar groups remain unresolved.

The feature histogram selects seed modes, not final pixel labels. Its bin width
is min(4, color_tolerance / 4), with exact feature bins for tolerances below 1
(reported as histogram_bin_width=0). This prevents histogram quantization from
merging colors at a finer requested tolerance.
Assignment operates in chunks. Color modes are capped explicitly and any overflow
makes the count unresolved; it does not silently report an exact capped count.
Runtime scales with image area, fragment count and mode count; no real-time
throughput guarantee is made.

| Constructor parameter | Default | Units / interpretation |
|---|---:|---|
| `background_segmenter` | None | New default RGBDBackgroundSegmenter when omitted. |
| `color_tolerance` | 18 | Euclidean distance in 255-scaled normalized RGB; not Delta E. Larger merges more colors. |
| `color_margin` | 4 | Required nearest-vs-second-nearest distance advantage. |
| `min_object_area` | 64 | Total retained visible pixels per color identity, inclusive. |
| `min_fragment_area` | 8 | Minimum 8-connected visible fragment area, inclusive. Smaller fragments remain rejected. Must be <= min_object_area. |
| `min_depth_support_fraction` | 0.1 | Minimum fraction with positive depth foreground outside the threshold margin, checked per group AND per retained fragment. |
| `exclusion_guard_px` | 2 | Square dilation radius around arm/exclusion/high-separation masks, 0..64. Can remove real adjacent cloth. |
| `max_reference_separation_m` | None | Optional positive upper empty-current optical-axis Z limit in metres. No universal arm height is assumed. |
| `max_color_groups` | 32 | Mode computation budget, 1..128. Exceeding it is unresolved. |
| `min_brightness` | 24 | Minimum maximum RGB channel value, in 8-bit encoded RGB. |
| `min_saturation` | 0.12 | Minimum median `(max-min)/max` for a group to avoid achromatic ambiguity. |
| `stability_fraction` | 0.2 | Refit at tolerance times 0.8 and 1.2 by default; differing counts block acceptance. Not a statistical interval. |

Pixel area thresholds scale quadratically with resolution. Set these below the
smallest visible portion that should count, while keeping residual speckles below
them. Multiple small fragments can collectively support a group if each meets the
fragment threshold. Pieces smaller than that threshold are deliberately ignored;
cropped objects with insufficient visible support may therefore be missed.

Tune registration, depth scale and actual invalid values first. Next tune the
background segmenter on empty scenes with changing shadows, then the exclusion
mask/upper depth band, area thresholds, and finally color tolerance. Use independent
representative captures for validation rather than tuning and evaluating on the
same frames.

## Noise and arm-residue tuning: small reproducible cases

Run `python examples/example_object_count_noise.py` for eight deterministic
160×220 synthetic cases using the direct-mask API. The example prints strict JSON,
including retained target/arm pixels, and performs no file or robot I/O. The same
controls apply to `detect()`. These are operating-limit examples, not camera
accuracy measurements or recommended deployment thresholds.

| Constructed case | Settings and measured outcome |
|---|---|
| Arm mask misses a 3-pixel perimeter | Guard 2 leaves 236 arm pixels and count 2; guard 3 removes the residue and gives count 1. |
| Real 2×40 strip beside a correctly masked arm | Guard 0 retains 80 target pixels and count 1; guard 2 removes all of them and gives count 0. |
| Eight separated 3×3 same-color residuals | Fragment minimum 8 retains all 72 residual pixels as an extra group; minimum 10 rejects them. A real object fragmented this way would also be lost. |
| Target separation 200 mm; arm separation 20 mm | Upper limit 150 mm returns count 1 for the wrong identity: zero target pixels and all 1600 arm pixels remain. |
| 400 target pixels plus 400 same-color residue pixels | Count stays 1 while half the returned group is residue. Numerically correct count does not establish a correct mask. |

Tune in this order:

1. Validate registration, depth scale/invalid values and empty-reference stability
   on repeated actual captures. Keep separate tuning and held-out frames.
2. Measure arm-mask under-coverage and target clearance. Sweep guards 0/1/2/3/4/6
   and choose the smallest that removes measured residue without deleting needed
   target support. If these requirements overlap, improve the arm mask.
3. Enable the upper camera-Z limit only when the tallest valid target plus depth
   uncertainty lies below the lowest arm minus uncertainty. Avoid exact threshold
   equality. Near-table arms and tall targets can make a safe gap impossible.
4. Sweep fragment minima 1/4/8/16 and aggregate minima 16/32/64/128 jointly at your
   resolution. Preserve crop/occlusion remnants. Each fragment must pass the first
   threshold; same-color fragments are then summed for the second. Coherent
   residue can pass both, including thin diagonal chains under 8-connectivity.
5. Calibrate depth threshold/margin to measured noise and actual thin targets.
   Compare support fractions .1/.25/.5. Just 10% coherent false support can meet
   the default .1; raising it also rejects genuine weakly observed thin objects.
6. Inspect lost target pixels, retained arm pixels, per-group identity, rejected
   and excluded masks, and abstentions, not just count. Preserve upstream and
   cleanup uncertainty. Shape cleanup is not semantic arm/noise removal.

When arm residue overlaps real targets in color, depth, and visible size, these
thresholds cannot distinguish them. Better arm segmentation, robot geometry,
temporal evidence, or different sensing is needed rather than aggressive filters.

## Important limits and counterexamples

- **The distinct-colors assumption is stronger than unequal RGB values.** Colors
  must be separated relative to noise, illumination variation and the configured
  tolerance. Two nearly equal colors can merge as one hypothesis. No threshold
  can reliably separate arbitrarily close colors from natural within-object color
  variation.
- **Brightness-only differences are non-identifiable under the shading model.**
  `(120,80,40)` and `(240,160,80)` have identical normalized color. Two materials
  of those colors and one material under half illumination can produce identical
  RGB-D. The counter returns one color-group hypothesis under its shading
  assumption, even if there were physically two objects. White/gray/black groups
  are explicitly ambiguous. Do not interpret an accepted count as resolving this
  fundamental ambiguity.
- Normalized encoded RGB is invariant only to proportional scaling in encoded
  RGB, not to arbitrary real light, sensor gamma, additive ambient illumination,
  specular highlights, clipping, white-balance changes or colored shadows.
  Significant changes can split one material or merge two. A hue/saturation model
  cannot remove these ambiguities either.
- Thin cloth without reliable depth change and a colored table shadow can produce
  the same observations. Such color groups are returned as candidates with
  `count=None`. A dark neutral table shadow can likewise remain ambiguous. Real
  shadows accompanied by coherent depth error may still pass depth support.
- The upper band is **camera-Z difference**, not calibrated table-normal height.
  It can remove legitimate tall folds or lifted objects. It works for arms only
  if the arm and target depth ranges separate. An unmasked arm near the table,
  or residual arm with target-like color/depth/area, can still be counted.
- Global same-color merging assumes fragments share identity. Their connected
  fragment count is reported separately. Occluded, cropped, below-threshold or
  completely hidden objects are not reconstructed. No finite bound on hidden
  physical objects follows from this result.
- A smooth multicolor gradient, pattern, or large highlight violates the
  near-solid-color assumption. Mode-count stability and spread checks flag some
  cases, not every possible violation.
- Alignment is a declaration, not a verified measurement. Color/depth misalignment,
  camera motion or a stale empty reference can create false groups. A declared
  unchanged background-check region can detect some reference failures but cannot
  certify the rest of the image.

The regression suite checks these contracts and includes explicit non-identifiable
counterexamples. It does **not** measure accuracy on actual deformable-object
captures or establish deployment readiness.
