# RGB-D mask and pick-candidate tuning

This guide covers `RGBDBackgroundSegmenter` and `OverheadPickCandidateDetector`.
See the [mask contract](RGBD_BACKGROUND.md), [pick contract](PICK_CANDIDATES.md),
and [parameter index](TUNING.md). Pixel clearance is not a gripper dimension;
these APIs do not determine material properties, finger access, or grasp success.

## Run the diagnostic workflow

From the repository root with the optional OpenCV dependency installed:

```bash
python -m pip install -e '.[debug]'
mkdir -p debug-output
python examples/example_rgbd_pick_tuning.py --output-dir debug-output/run-01
```

Without an input file, this processes a generated scene containing a hole, a
separate object inside the hole, a depth spike and missing depth. It is not a
real-image accuracy evaluation. The output directory must be new and its parent
must exist; existing files are not overwritten. Use `--help` for arguments.

| Output | Purpose |
| --- | --- |
| `parameters.json` | All effective constructor settings; reusable as `--config` |
| `summary.json` | Units, color order, selected contour ID, status, best point and stage counts |
| `segmentation.png` / `.json` | Foreground, uncertainty, missingness, color/depth evidence and reference check |
| `outline.png` / `.json` | Contour IDs and holes; check which object is selected |
| `pick.png` / `.json` | Selected component, support, candidate regions, representative points and rejection reasons |
| `pixel.json` | Masks, scores and support at the requested pixel, or the best point; null if neither exists |
| `arrays.npz` | Result arrays prefixed `seg_` / `pick_`, diagnostics prefixed `seg_diag_` / `pick_diag_` |

Change one setting at a time and rerun the same input into a new directory.
Config JSON accepts only `segmenter` and `pick` objects; omitted settings use
implementation defaults. This is a syntax example, not a deployment preset:

```json
{
  "segmenter": {"background_luma_tolerance": 80.0},
  "pick": {"local_depth_tolerance_m": 0.004}
}
```

Save it as `tuning.json`, then run:

```bash
python examples/example_rgbd_pick_tuning.py --config tuning.json --output-dir debug-output/run-02
python examples/example_rgbd_pick_tuning.py --inspect-x 85 --inspect-y 75 --output-dir debug-output/spike
```

The second command inspects the isolated near-depth spike. Supply both inspection
coordinates to inspect any pixel, even when no candidate exists. Array indices
are `[y, x]`; API points are `[x, y]`.

## Use recorded inputs

Register all four arrays to the same grid before saving them:

```python
import numpy as np
np.savez_compressed(
    'pair.npz', color=current_color, depth=current_depth,
    background_color=empty_color, background_depth=empty_depth,
    # Optional same-grid bool masks:
    # depth_valid_mask=current_valid,
    # background_depth_valid_mask=empty_valid,
    # background_check_mask=known_unchanged_background,
)
```

```bash
python examples/example_rgbd_pick_tuning.py --input pair.npz --color-order RGB --depth-scale-m 0.001 --output-dir debug-output/real-01
python examples/example_rgbd_pick_tuning.py --input pair.npz --color-order RGB --depth-scale-m 0.001 --config tuning.json --contour-id 0 --output-dir debug-output/real-02
```

Choose the actual ID from `outline.json`; 0 is only an example. Without an ID,
the largest outer polygon area wins, with the smallest ID breaking ties. This
area includes holes. `outline.component_mask(id)` then selects the original
visible 8-connected component without filling holes. Touching objects may share
a component. Recompute the outline whenever its input mask changes.

Real inputs require explicit `--color-order` and `--depth-scale-m`. The example
declares registration to the API but does not perform it. NPZ loading disables
pickle. The generated scene uses fixed RGB and .001 metres/count.

## Check inputs before tuning

1. **Registration and units:** colors must be uint8 HxWx3; depths must be real
   numeric HxW, with H and W at least 2. Match pixel coordinates, resolution,
   crop and rectification across all four images. Use a fixed camera/background
   and an independent empty reference. Pick detection assumes an overhead view.
   Depth is optical-axis Z, not ray range. Use .001 for mm/count or 1 for metres,
   including floating-point arrays; units are never inferred from dtype.
2. **Missingness and domain:** inspect nonpositive, nonfinite and sentinel depth
   and validity masks. Do not replace missingness with zero difference. Mark
   interpolated or unreliable pixels invalid. Restrict selection with `work_roi`.
   A reference-check ROI must be known unchanged background, excluding objects,
   shadows and moving surfaces.
3. **Evidence:** inspect color and depth foreground independently. Estimate depth
   difference noise from repeated empty captures before setting
   `depth_threshold_m`, then tune chroma/luma. Using the current image as its own
   reference hides differences; it does not validate a background model.
4. **Components and uncertainty:** inspect mask, contour IDs, holes and
   `seg.uncertain_mask`. Availability and classification uncertainty are distinct.
5. **Pick support:** inspect `object_pixels`, `search_pixels`, `pair_valid_pixels`
   and `supported_pixels` with their masks. These are not a simple subtraction
   sequence: pair-valid pixels need not be a subset of the search domain.
   Resolve units, missingness, support and clearance before lowering elevation.
6. **Elevation, size, ranking:** tune elevation, area and radius only after
   support exists; tune the nearest-depth band last. Compare `rejected_regions`,
   depth, support and clearance. Fewer unknown pixels alone is not improvement.

Compare empty, target-present, shadow/lighting-change and missing-data cases with
one fixed configuration. On resizing, review pixel lengths, windows and areas.
Metric sensor thresholds do not change merely because of resizing, but a window
covering wider-spaced pixels may span larger depth changes on slopes.

## Segmenter settings

Numbers must be finite scalar values convertible to float, excluding bool.
Integer settings require Python int, excluding bool; boolean settings require
True/False. Invalid values raise `ValueError`. Adjustment directions below do
not promise monotonic improvements in accuracy.

| Setting | Default and range | Adjustment |
| --- | --- | --- |
| `background_chroma_tolerance` | 20; >0, encoded 8-bit Lab | Increase to ignore more chroma variation; decrease to detect weak differences and more color noise |
| `background_luma_tolerance` | 60; >0, encoded Lab | Increase to ignore brightness/shadows, at the cost of brightness-only targets |
| `background_score_margin` | .2; 0..1 | Widen uncertainty around ellipse radius 1; also changes evidence confirmation/background agreement and strict chroma seeds |
| `depth_threshold_m` | .005; >0 metres | Increase to reject smaller changes; decrease to admit thinner objects and noise. Also sets the negative farther-change threshold |
| `depth_margin_m` | .001; >=0 and < depth threshold | Widen threshold uncertainty without changing raw depth foreground; may exclude more downstream pick pixels |
| `fusion_mode` | `'union'`; `'union'`, `'depth_required'`, `'evidence'`, `'color_seeded'` | Choose evidence policy, not a confidence level; see the API's fusion rules |
| `missing_depth_policy` | `'color'`; `'color'` or `'unknown'` | Keep the chosen mode's color candidates with uncertainty, or require depth. Depth-required always requires depth |
| `min_object_area` | 1; int >=1 pixels | Remove smaller 8-connected components with uncertainty; does not fill holes or split touching objects |
| `background_compensate` | False; bool | Apply global median Lab offset from at least 25 valid known-background check pixels; does not fix local shadows or registration |
| `background_check_p95` | 20; >0, unweighted encoded-Lab distance | Increase to accept larger corrected check residuals; inspect ROI, lighting and registration first. Inactive without a check region |
| `background_max_offset` | 40; >0, Lab vector norm | Increase to allow larger global correction; may hide bad references. Inactive without compensation |
| `strong_depth_ratio` | 3; >=1 | Evidence only: increase to require stronger depth-only changes; default strong threshold is 15 mm |
| `depth_support_radius_px` | 1; int 1..5 | Widen the square neighborhood; may lose narrow shapes. Changing the denominator makes effects nonmonotonic |
| `depth_support_fraction` | .25; (0,1] | Require more agreeing neighbors for evidence depth-only selection and seeded growth; lower values preserve thin detail but admit noise |
| `depth_support_tolerance_m` | .003; >=0 metres | Widen agreement in neighbor/center dZ to tolerate slopes and noise; zero means exact floating-point equality |
| `color_relaxation_ratio` | .5; (0,1] | Seeded growth only: lower values admit weaker color changes; seed threshold is unchanged |
| `color_growth_max_steps` | 2; int 0..20 | Seeded growth only: maximum 8-neighbor steps from an initial seed; zero returns seeds only |

Color foreground uses `score=(dC/Tc)^2+(dL/Tl)^2 > 1`; the uncertainty margin
applies to `sqrt(score)`, not score. This is neither a probability nor standard
Delta E. Depth foreground requires `dZ=empty-current > T`; farther change is
`dZ < -T`. Equality does not pass. A failed color check does not stop depth
evaluation or trigger border-color guesses. Evidence mode then requires strong,
locally supported depth; color-seeded mode cannot produce seeds.

## Pick settings

The same numeric type rules apply. See [selection rules](PICK_CANDIDATES.md#selection-rules-and-the-two-depth-roles)
for the exact support and ranking contracts.

| Setting | Default and range | Adjustment |
| --- | --- | --- |
| `min_elevation_m` | .01; >0 metres | Require greater center and local-median reference separation after subtracting reference error |
| `reference_error_m` | .002; >=0 metres | Increase the assumed error allowance; effective separation minimum is elevation plus error. Error is not estimated automatically |
| `boundary_guard_px` | 3; >=0 pixels | Increase clearance from outline, holes, missingness, uncertainty and image border; may remove the entire search interior |
| `local_radius_px` | 2; int 1..10 | Widen the `(2r+1)` square; suppress isolated spikes but mix more slopes, creases and boundaries |
| `min_support_fraction` | .8; (.5,1] | Require more of the full square to agree with both medians; missing/outside-component samples remain in the denominator |
| `local_depth_tolerance_m` | .003; >0 metres | Increase tolerance around current-Z and separation medians; admits slopes/noise and potentially spikes |
| `min_region_area` | 25; int >=1 pixels | Reject small supported elevated regions; separate from upstream object-area filtering |
| `min_region_radius_px` | 2; >=0 pixels | Require wider candidate interiors for representatives, including in large narrow regions; not a gripper radius |
| `nearest_depth_band_m` | .003; >=0 metres | Allow representatives farther than the minimum supported local Z to gain clearance; zero allows only the minimum |

Guard and radius tests use strict `>` distance to excluded pixel centers. An edge
foreground pixel has distance 1; these are not subpixel edge distances. Support
uses only trusted samples in the selected component, but divides by the entire
square. The center must also agree with both medians. Lowering elevation cannot
restore upstream-excluded or uncertain pixels. Regions are eligible by reference
separation but ranked by representative local current Z, then larger area, then
y/x. Within a region, clearance wins among depth-band-qualified points, with
topmost/leftmost tie-breaking. This is not maximum-difference ranking.

## Detection masks and optional inputs

Both detectors require positive finite `depth_scale_m`, also for float arrays,
and explicit registration (`registered_to_depth=True` or
`registered_to_mask=True`). Segmentation requires `color_order='RGB'` or `'BGR'`
for both colors. Registration flags are attestations, not alignment operations.

| Input | Contract |
| --- | --- |
| `color_valid_mask`, `background_color_valid_mask` | Segmenter bool HxW; omitted means all color available. Black does not automatically mean missing |
| `depth_valid_mask` | Both APIs, bool HxW; automatic invalid-depth checks still apply when omitted |
| `background_depth_valid_mask` / `background_valid_mask` | Segmenter / pick reference-depth validity; note the different names |
| `work_roi` | Segmenter bool HxW; omitted means full image. Outside it object, observed and uncertain masks are False |
| `background_check_mask` | Segmenter known-unchanged bool HxW; may lie outside ROI. At least 25 jointly valid color pixels when supplied or compensating; does not prove an empty reference |
| `contour_id` | Pick int ID, not bool; default largest outer polygon. Unknown explicit IDs raise ValueError; empty outline with no ID gives no_object |
| `segmentation_uncertain_mask` | Pick bool HxW; True excludes both support and selection. Omitted means trust the supplied object mask; not a sensor-validity substitute |
| `invalid_values` | Finite real 1D sequence, default empty; applies the same sentinels to both depths in original count units |
| `diagnostics` | Bool, default False; retain dense diagnostic arrays. Pick point arrays are always retained |

The example NPZ accepts the segmenter's optional masks and `invalid_values`.
It passes `seg.depth_valid_mask` (pair validity) and `seg.uncertain_mask` to pick.
Recompute the whole pipeline if the reference changes.

## Read debug output

```python
seg = segmenter.detect(color, depth, background_color=empty_color,
    background_depth=empty_depth, color_order='RGB', depth_scale_m=.001,
    registered_to_depth=True, diagnostics=True)
outline = OuterContourDetector().detect(
    seg.object_mask, observed_mask=seg.outline_observed_mask)
pick = picker.detect(depth, outline, background_depth=empty_depth,
    depth_scale_m=.001, registered_to_mask=True,
    depth_valid_mask=seg.depth_valid_mask,
    segmentation_uncertain_mask=seg.uncertain_mask, diagnostics=True)
image = seg.render_debug(color, color_order='RGB')
seg.save_debug('segmentation.png', color, color_order='RGB')
pick.save_debug('pick.png', color, color_order='RGB')
```

The parent directory must exist. Existing PNGs are rejected unless explicitly
using `overwrite=True`. Common display settings are `mask_alpha=.35` (finite
0..1), `panel=True` (bool), and `max_lines=32` (int 1..100). Without color, provide
`depth` and `depth_scale_m` for percentile-scaled visualization; this changes
only display. `color_order` applies to rendering input and output; PNG saving
converts the order as needed. Computation uses the original grid even if display
is reduced. Generic panels show at most 13 tiles; inspect omitted arrays directly.
Segmenter diagnostic heatmaps still appear with `panel=False`.

- Inspect `stages['color_reference']` and `color_usable_mask` first: numeric color
  maps remain available even when the reference is rejected.
- `background_chroma_distance` / `background_luma_distance` are dC/dL.
  `background_combined_score` is the ellipse score (heatmap saturates above 4).
  `background_distance` is a separate unweighted, corrected Lab distance.
- `depth_difference_m` is signed empty-current separation; its heatmap saturates
  outside ±2T, with missing NaN in magenta. Numeric values are retained.
- `background_valid_mask`, `background_check_mask`, and `color_usable_mask`
  distinguish jointly available color, declared check region and usable evidence.
  `background_photometric_clipped_mask` flags RGB values <=1 or >=254;
  `background_score_saturated_mask` flags numeric saturation, not high certainty.
- `color_foreground_mask` / `depth_foreground_mask` separate evidence sources.
  `conflict_mask` is color foreground at farther depth. `farther_mask`,
  `depth_threshold_margin_mask`, `color_only_mask`, `filtered_mask` and `work_roi`
  distinguish rejection, uncertainty and processing-domain effects.
- `object_mask` and `uncertain_mask` can overlap. Do not use the inverse of all
  uncertainty as outline availability: this creates artificial cuts. Use
  `outline_observed_mask`, and propagate uncertainty separately to pick.
- For pick, inspect selected `object_mask`, then `excluded_mask`, `unknown_mask`,
  `supported_mask`, `eligible_elevation_mask` and `candidate_mask`. Support alone
  does not establish elevation/area/width. Not every candidate-mask pixel meets
  the representative-point radius condition. Excluded and unknown may overlap.
- Pick `pair_valid_mask` includes component/uncertainty restrictions;
  `clearance_px` measures distance to excluded centers. `local_depth_m` and
  `local_separation_m` are medians, not filled measurements. `support_fraction=0`
  can mean unevaluated: also inspect NaN local maps and masks.
- `candidate_points_xy` and `selected_point_xy` show all representatives and the
  best point. Pick numeric maps are not automatically rendered as heatmaps.

Inspect a numeric pick value directly:

```python
with np.load('debug-output/run-01/arrays.npz', allow_pickle=False) as a:
    x, y = 85, 75
    print(a['pick_diag_local_depth_m'][y, x])
    print(a['pick_diag_support_fraction'][y, x])
    print(a['pick_unknown_mask'][y, x])
```

`pick.candidates` records `point_xy`, `area_px`, `current_depth_m`,
`local_depth_m`, `reference_separation_m`, `local_separation_m`,
`local_support_fraction`, `clearance_px`, `region_clearance_px`,
`nearest_supported_depth_m`, `id`, `rank` and `grasp_suitability`.
IDs need not equal ranks or list indices; `pick.point_xy` is the best point.

## Troubleshooting

| Symptom | Inspect and adjust |
| --- | --- |
| Shadows/lighting enter the mask | Check dC/dL, color-only evidence and reference check. Increase luma tolerance for brightness changes; consider global compensation only with a valid check region. Depth-required loses thin targets |
| Same-color target missing | Check units, pair validity and dZ. Lower T only after measuring noise. Identical background/target evidence cannot be separated by thresholds |
| Thin target missing | Inspect color score, fusion and object-area filtering. Union can retain color-only candidates, but their uncertainty excludes them from trusted pick support |
| Noise becomes a pick | Compare center and medians. Try tighter tolerance or greater support/window/area/radius one at a time. Coherent bias still passes; check the reference and sensor |
| Points near holes/boundaries | Increase guard/radius; widen nearest-depth band to prefer more clearance. Candidates may disappear if too narrow |
| No candidate or unknown | Follow segmentation, component, reasons, support/elevation and rejected regions. Lower elevation cannot fix absent support; inspect slope, missingness and window size |
| Many small candidates | Inspect holes in support and elevation masks. Area/radius can reject fragments; looser support can connect them but also admit noise. Select the intended contour |
| Point differs from nearest raw pixel | Expected when supported median Z and clearance beat an isolated minimum. A narrower depth band prioritizes proximity, not grasp success |

Segmenter `candidate` still permits uncertainty elsewhere; `unresolved` means no
selection but suppressed/deferred candidates remain. `no_change_candidate` does
not prove absence. For `insufficient` or `empty_domain`, inspect validity, policy
and ROI. Pick `no_object` points upstream; `unknown` requires checking trusted
interior, missingness and consistency; `no_candidate` means evaluated regions
failed elevation/area/width. Even `candidate` leaves grasp suitability unverified.

## Evidence mode

Use `fusion_mode='evidence'` to compare color confirmation, depth strength and
local depth support. See [decision rules and codes](RGBD_BACKGROUND.md#evidence-fusion)
for the authoritative branches. This is not fixed-width erosion, a probability,
or a temporal-stability guarantee.

1. Verify registration, units and reference, then set depth and color thresholds.
2. Compare union and evidence on identical inputs; inspect `suppressed_mask`,
   `deferred_mask` and decision reasons, including unresolved regions in a
   result whose overall status is candidate.
3. If shallow same-color targets disappear, consider a lower `strong_depth_ratio`
   or union with explicit uncertainty. Color agreement does not prove background.
4. If narrow same-color detail disappears, inspect support fractions before
   lowering the fraction/radius. Defaults need 2 of 8 neighbors: rectangular
   corners with 3 survive, but a one-pixel-line endpoint with 1 fails. Color-
   confirmed features do not need this support branch.
5. If inconsistent depth-only candidates remain, increase support or reduce
   tolerance. Strong uniform reference bias still passes; acquire/check an
   independent reference rather than assuming local agreement proves correctness.

Accepted depth-only and chroma-only regions remain uncertain and therefore do not
provide trusted pick support. Do not clear uncertainty to force a pick. Luma-only
thin targets may be deferred as indistinguishable from shadows; shadow plus depth
bias can pass both modalities. Clipped/missing/rejected color cannot confirm or
suppress depth. Color registration artifacts may remain chroma-only candidates.

Use `seg.outline_observed_mask` for outlines: observed minus suppressed/deferred,
not minus all uncertainty. Unknown cuts can split contours. Legacy modes have no
suppressed/deferred pixels, so this equals their observed mask.

With diagnostics, the fusion panel uses green for both modalities, blue for
strong supported depth only, cyan for chroma only, red for suppression, purple
for deferral and orange for area filtering. Inspect `fusion_decision` and
`stages['fusion']['decision_reasons']`, plus `depth_support_fraction`,
`strong_depth_mask`, `locally_supported_depth_mask` and reliable-color masks.
Accepted masks reflect final area filtering.

```bash
python examples/example_rgbd_fusion.py --output-dir debug-output/fusion-01
```

For recorded input, use `{"segmenter":{"fusion_mode":"evidence"}}` in the
existing tuning config. The example saves reason/support arrays and
`seg_outline_observed_mask`. Generated examples demonstrate rules and failures,
not real-pair precision/recall.

## Color-seeded mode

Use `fusion_mode='color_seeded'` for strict chroma seeds with bounded weak-color
edge growth. See [growth rules and diagnostics](RGBD_BACKGROUND.md#bounded-color-growth).
Start with `color_growth_max_steps=0` to inspect seeds, then try 1–2 and compare
`accepted_growth_mask` against the original image. Check registration, missing
depth and dZ support before lowering `color_relaxation_ratio`.

No seeds means no target, even with strong depth. Growth cannot jump gaps or
reset its distance bound at new pixels. All additions remain uncertain; pass that
uncertainty to pick. The three depth-support settings gate additions, not seeds;
`strong_depth_ratio` is unused. A real hole with erroneous color and depth may
still be filled within the bound. False seeds from colored shadows or registration
are not removed by bounding growth. Detail/topology preservation is not guaranteed.

`growth_step` is -1 for unreached, 0 for seeds and 1..N for additions. Seed/step
maps precede area filtering; accepted-growth masks follow it. Run
`PYTHONPATH=src python examples/example_rgbd_seeded.py` for the minimal example,
or select this mode in the NPZ example config.

## Fixed settings across captures

The segmenter and inward corner offsets do not estimate optimal settings per
capture. Fixed settings require a defined operating range, aligned reference and
consistent capture conditions. **Overlapping target and nuisance observations
cannot be separated by threshold tuning.** The 5/80 mm depth band and two-pixel
growth examples are not calibrated deployment recommendations.

### Depth-band separation

For an upper camera-Z difference bound U, let `target_max` be the greatest true
target difference over the operating range and `other_min` the smallest unwanted
near-surface difference. With independent worst-case error bounds, separation
requires:

```text
target_max + error_target <= U < other_min - error_other
```

This addresses only the upper bound; lower thresholds, missingness and color
still matter. It is conservative if the extrema/errors cannot coincide. If
observed target and nuisance ranges overlap, no shared upper bound exists. For
example, target/occluder differences of 100/130 mm in one capture and 20/60 mm in
another are individually separable but have no common upper limit. The 80 mm
example loses the first target and admits the second occluder. Folds, lifting
and sag affect target extent, not just material thickness.

Camera-Z difference is not table-normal height. For parallel planes with normal
`n` directed from camera toward table, table `n dot X=d`, raised surface
`n dot X=d-h`, and ray `r=((u-cx)/fx,(v-cy)/fy,1)`, positive `n dot r` gives
`dZ=h/(n dot r)`. At the center ray, 60 mm normal height becomes about 84.85 mm
at 45 degrees. This derivation's normal is opposite the above-table-positive
`TablePlane` convention. Real surfaces need not be planar. An equal Z translation
of an ideal frontal current/reference pair cancels in the difference, so distance
alone does not always require a new bound; pixel scale, sensor error, view and
correspondence can nevertheless change.

For pose-independent height, use calibrated intrinsics, a verified current table
plane, `CameraIntrinsics.deproject` and `TablePlane.height`. This still cannot
identify unwanted objects at the same height. Use independently justified ROIs,
tracking, semantics or another view where needed; exclusion does not recover
hidden target material. See the [depth-band workflow](RGBD_BACKGROUND.md#dark-shadows-and-much-nearer-objects).

A lower bound L requires background errors <=L and desired observed target
differences >L across captures. A 4 mm background error and 2 mm target cannot
be separated this way. Local support can accept coherent bias: a uniform 7 mm
reference error passes a 5 mm threshold. `background_check_mask` validates and
compensates **color only**, not depth drift. `depth_margin_m` marks uncertainty;
it is not hysteresis. Neither support nor `strong_depth_ratio` guarantees truth.
Refresh references/calibration when camera or table geometry changes, measure
repeated empty-depth errors, and monitor known unchanged regions in the
application. The API does not validate temporal stability or depth reference drift.

### Color seeds and projected scale

Near the strict seed threshold, slight color variation can remove an entire
unseeded object. For example, with background RGB (140,140,140) and default
thresholds, a uniform target changing from (110,145,182) to (110,145,183) crosses
approximately dC=23.19 to 24.08, across the strict `dC > 24` boundary in encoded
8-bit Lab. This illustrates a threshold discontinuity, not a camera-specific
failure frequency. More growth cannot recover zero seeds; lowering the relaxation
ratio does not change seed eligibility. Growth limits constrain distance from
seeds, not the number or area of false seeds elsewhere.

Fix exposure, white balance, lighting, background and pose where possible; verify
seed margins on independent captures. Global Lab compensation does not fix local
shadows, clipping, motion or registration. Do not loosen reference-check bounds
merely to accept incompatible captures.

| Pixel setting | Effect of capture variation |
| --- | --- |
| `color_growth_max_steps` | A physically identical weak edge can project wider than the allowed growth distance |
| `depth_support_radius_px` / fraction | Small projected shapes and missing samples reduce support; increasing radius is not monotonic |
| `min_object_area` | Projected area changes approximately with the square of image scale |
| `corner_window_px` / `clip_margin_px` | Direction support and distance from unobserved boundaries depend on projected shape and observation mask |
| `corner_interior_points(offset_px)` | Exact pixel displacement or unavailable, not millimetres or a grasp clearance guarantee |

For a frontal local plane, lateral displacement is approximately
`offset_px * Z / fx`; twice the depth gives twice the physical displacement for
the same pixel offset. Slopes/curves need a different conversion. Corner offsets
are deterministic relative to the input contour; changing contours move them,
and holes, narrow regions or unknown areas can make them unavailable. See the
[offset contract](OUTLINE.md#separate-inward-offset-corner-positions).

Units, color order, sentinels, validity and registration are acquisition facts,
not tuning parameters. Fusion/missingness policies define what remains unknown;
do not switch them per image merely to match a desired answer.

### Validate one configuration

1. Define distance/pose, target color/thickness/lifting, lighting and occluder
   operating ranges. Acquire matching references/calibration for each geometry.
2. Collect independent current/empty pairs and repeated empty scenes. Annotate
   true contours separately from unknown regions and measure color/depth error.
3. Separate tuning and held-out captures. Freeze settings and measure worst-case
   target loss, shadow/occluder leakage, seed loss, corner variation and unknown
   rates rather than retuning each image.
4. If there is no separating margin across the range, improve acquisition or add
   evidence before adding parameters. Insufficient real data is not validation.

Synthetic tests establish API behavior, including counterexamples; they do not
establish repeated real-capture stability or robot safety. Constructed references
on recorded arrays likewise test input handling, not genuine empty/current
accuracy. Independent registered real-pair accuracy and physical grasp success
remain unvalidated. Broad elevated flat sheets can pass, while thin creases or
steep slopes can fail support/width requirements.
