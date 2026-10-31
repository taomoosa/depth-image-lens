# Overhead geometric pick candidates

For all parameter ranges, direction of adjustment, symptom-based troubleshooting,
and a runnable NPZ/debug workflow, see [the tuning guide](RGBD_PICK_TUNING.md).

`OverheadPickCandidateDetector` finds supported, elevated, near-camera regions
inside exactly one visible object component. It returns candidate region masks,
representative `[x, y]` pixels and reasons/measurements. **It does not decide that
an object can be pinched or guarantee a successful grasp.** It requires the same
optional OpenCV dependency as the outline and RGB-D background detectors.

This complements `DepthSurfaceDetector`: that class inspects elevation and local
nonflatness in fixed cells; this detector selects locally supported pixels and
region representatives with clearance requirements. It neither requires a fold
nor interprets elevation as multilayer structure. Existing detector defaults are
unchanged.

```python
from depth_image_lens import (
    RGBDBackgroundSegmenter, OuterContourDetector, OverheadPickCandidateDetector,
)

seg = RGBDBackgroundSegmenter().detect(
    current_color, current_depth,
    background_color=empty_color, background_depth=empty_depth,
    color_order='RGB', registered_to_depth=True, depth_scale_m=.001,
)
outline = OuterContourDetector().detect(
    seg.object_mask, observed_mask=seg.observed_mask,
)
# None selects the largest polygon area (smallest contour ID breaks ties).
# Pass an explicit ID to select another visible object component.
chosen_id = outline.largest_contour()['id'] if outline.contours else None
pick = OverheadPickCandidateDetector(
    min_elevation_m=.01, reference_error_m=.002,
    boundary_guard_px=3, local_radius_px=2,
    min_support_fraction=.8, local_depth_tolerance_m=.003,
    min_region_area=25, min_region_radius_px=2,
    nearest_depth_band_m=.003,
).detect(
    current_depth, outline, contour_id=chosen_id,
    background_depth=empty_depth, depth_scale_m=.001, registered_to_mask=True,
    depth_valid_mask=seg.depth_valid_mask,
    segmentation_uncertain_mask=seg.uncertain_mask, diagnostics=True,
)
if pick.point_xy is not None:
    print(pick.point_xy, pick.candidates[0])
# No files are written during detection. Optional explicit output:
# pick.save_debug('pick-candidates.png', current_color, color_order='RGB')
```

Run `python examples/example_pick_candidate.py` in an installed checkout
(or use `PYTHONPATH=src`). Add `--output /existing/path/new-file.png` for overlays.
The example retains a hole and excludes its separate nested object and an
isolated nearer depth spike. Its generated scene is not real-image evaluation.

## Assumptions and inputs

The camera looks down from above; current and empty-background optical-axis Z
images use the **same grid, view, crop and metres/count scale**. Explicit
`registered_to_mask=True` declares this contract; the function never aligns,
resizes, operates a camera or infers units. `depth_scale_m` is required: .001 for
millimetres/count, 1 for floating-point metres. A reference image is required.
The background scene must be unchanged and empty of the target. Camera tilt,
reference motion, parallax/alignment errors and depth biases invalidate the
simple reference interpretation. Even overhead, an uneven baseline means camera
Z separation is not automatically physical clearance under the material.

Color participates upstream in `RGBDBackgroundSegmenter` to define the object
mask and segmentation uncertainty; pick geometry does not reclassify color.
Outline must be a fresh `OuterContourResult` for that mask and grid. Do not edit
its stored mask/contour records between extraction and selection.

`outline.component_mask(contour_id)` is now an installed method. It labels
`object_mask & observed_mask` with 8-connectivity and uses a raw contour seed,
without filling a polygon. Holes, nonconvex notches, separate nested objects and
other components are therefore excluded. An empty outline without an explicit
ID returns an empty mask; unknown IDs and malformed/out-of-grid seeds raise
`ValueError`. The old `examples.example_vertical_interior.contour_component_mask`
helper delegates to the method. Objects touching in the same 8-connected
component remain inseparable; IDs are identifiers, not array indices.

Both depths must be positive finite numeric HxW arrays matching the outline.
Nonpositive/NaN/Inf depth, sentinels in `invalid_values` (original count units),
and false `depth_valid_mask` or `background_valid_mask` are unavailable. Metric
conversion overflow/underflow is also unavailable. These bool masks concern
current/reference sensor support, not semantic object identity. A pair-valid
mask such as `seg.depth_valid_mask` may be passed as the current mask when using
the same depth/reference pair. Input arrays are never changed.

Optional bool `segmentation_uncertain_mask` excludes uncertain pixels from both
local support and candidate selection, with clearance around them. Pass the
upstream uncertainty when available; otherwise the supplied object mask is
assumed trustworthy. Missingness and uncertainty are never filled to make a
larger candidate region.

## Selection rules and the two depth roles

1. Extract the selected visible contour component. Exclude segmentation-uncertain
   pixels. Find the interior whose distance from an excluded pixel center exceeds
   `boundary_guard_px`. The image border and observation/ROI cuts also count as
   excluded boundaries.
2. Reject centers within that guard distance of a missing current/reference
   sample. For each remaining center use a `(2*local_radius_px+1)` square window.
   Only valid, trusted samples of this selected component contribute.
3. Compute median current depth and median `empty-current` separation in the
   window. More than half of the **entire square** (default 80%) must agree with
   both medians within `local_depth_tolerance_m`. Missing/outside samples count
   against support. The center itself must agree with both medians. This rejects
   isolated current-depth and reference-depth spikes and strongly mixed windows.
4. Both the center separation and its local median, minus `reference_error_m`,
   must reach `min_elevation_m`. Reference separation gates whether the surface
   is visibly above its baseline. Absolute current depth alone cannot establish
   this: a nearby background surface is not a raised object region.
5. Form 8-connected elevated regions. Reject those below `min_region_area` or
   with no pixel farther than `min_region_radius_px` from the region's excluded
   pixel centers. This width condition avoids choosing a long narrow strip
   merely because its total area is large.
6. Within each region's width-qualified pixels, find the smallest local median
   **current depth**. Among points within `nearest_depth_band_m` of it, choose
   greatest clearance (minimum of region clearance and valid/trusted clearance),
   then topmost/leftmost for a tie. Rank regions by representative local current
   depth ascending, then area descending, then y/x. The first representative is
   `point_xy`. Ranking is camera proximity, **not maximum background difference**;
   the two rankings can differ over an uneven background.

These medians are evidence summaries, not depth hole filling or new sensor
measurements. Returned points always correspond to valid observed selected
pixels. The candidate mask contains each accepted elevated region, including its
narrower edge pixels; only representative points are required to satisfy the
region-radius constraint. Local work is chunked to bound window temporaries.

## Settings

| Setting | Default | Units / contract |
| --- | ---: | --- |
| `min_elevation_m` | .01 | Positive baseline separation after error allowance |
| `reference_error_m` | .002 | Nonnegative assumed reference error allowance, metres |
| `boundary_guard_px` | 3 | Nonnegative clearance around contour, missingness and uncertainty |
| `local_radius_px` | 2 | Integer 1..10; default window is 5x5 |
| `min_support_fraction` | .8 | Consistent fraction of full square, strictly >.5 and <=1 |
| `local_depth_tolerance_m` | .003 | Positive tolerance around both local medians |
| `min_region_area` | 25 | Positive integer accepted area, pixels |
| `min_region_radius_px` | 2 | Nonnegative minimum interior distance for representative candidates |
| `nearest_depth_band_m` | .003 | Nonnegative tolerance from nearest supported local Z |

Pixel distances are Euclidean distance to the nearest excluded **pixel center**,
with an artificial one-pixel zero border outside the image. A boundary foreground
pixel therefore has distance 1. Both guard and region-radius comparisons are
strict `>`; these are not subpixel edge distances. Area and pixel settings depend
on image resolution and **do not encode gripper dimensions**. The error allowance
and default tolerances are starting values, not calibrated confidence bounds.

## Results and debug

- `state='candidate'`: one or more geometric candidates, even if other selected
  areas are unknown. `point_xy` is the first ranked point.
- `state='no_candidate'`: observed/evaluable search pixels produced no region
  meeting elevation, area and width settings. This is not proof that nothing
  can be physically picked.
- `state='unknown'`: no accepted candidate and missing/inconsistent evidence,
  segmentation uncertainty, or no interior with the requested clearance.
- `state='no_object'`: no visible contour component. `point_xy` is None whenever
  no candidate exists. Invalid input contracts raise `ValueError`.

`candidates` records one representative, area, measured/current local Z,
measured/local reference separation, local support fraction, input and region
clearances, rank and `grasp_suitability='unverified'`. IDs remain component labels,
not rank indexes. `metrics` reports counts and rejected regions with reasons.

Masks: `object_mask` is the selected component, `supported_mask` passes local
consistency, `candidate_mask` is accepted elevated regions, `excluded_mask` is
outside the trusted interior, and `unknown_mask` retains segmentation uncertainty
and unevaluable search pixels. Uncertain excluded pixels may appear in both
excluded and unknown masks. A region rejected for observed area/width limits is
not, by that rejection alone, unknown.

`render_debug`/`save_debug` show separate masks and candidate points. All-candidate and best-selected point arrays
are always recorded. With `diagnostics=True`, dense `depth_difference_m`,
`local_depth_m`, `local_separation_m`, `support_fraction`, `clearance_px`,
`pair_valid_mask`, and `eligible_elevation_mask` are also retained. `to_dict()`
provides JSON-safe settings, status, candidates and metrics; arrays stay separate.

## Counterexamples and limits

A broad, elevated, flat sheet can pass all geometric conditions with no accessible
pinch edge. The detector cannot infer friction, stiffness, available underside,
finger collision, object thickness, reachability, force or grasp success. These
need actual gripper geometry and physical validation. A thin crease or steep
ridge may fail the support or width conditions despite being pinchable. A
coherent sensor bias covering a whole patch can pass local consistency; isolated
noise rejection is not general noise immunity. A smooth slope can yield a
near-camera candidate without a distinct bump. No material/semantic recognition
is attempted.

Tests cover raised patches, nearest isolated spikes, reference spikes, mixed
boundaries, holes/nested objects, thin regions, missing depth, inconsistent
surfaces, a near-camera surface with insufficient reference separation, ranking
against a nonuniform baseline, and the flat-sheet counterexample. The checkout
has no suitable registered real RGB-D current/empty evaluation pairs; real-image
accuracy and physical pinch success remain unvalidated.

Limited checks on stored real depth arrays with constructed reference images
confirm input handling and exclusion behavior only. Some noisy/sloping recorded
surfaces return `unknown` under the default local tolerance and width settings,
even with an artificially positive reference separation. Inspect support and
clearance diagnostics before tuning; a rejected region is not proof of physical
unpickability. No independent overhead empty/current capture pair was available
for these checks.
