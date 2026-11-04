# Open template to partial exterior matching

`PartialOutlineMatcher` compares an ordered **open** target polyline with a
continuous arc of each visible closed exterior from `OuterContourDetector`.
For example, select the hem, armpit, cuff, shoulder and neckline on one side of
a correctly spread shirt. The target can cover about 30–40% of its perimeter.
The result describes that visible part only. A matching side can coexist with a
crumpled opposite side; the result is not a whole-shirt spreadness classifier.

```python
import numpy as np
from depth_image_lens import (
    OuterContourDetector, PartialOutlineMatcher, OutlineArticulation,
)

# Ordered XY image coordinates; do not repeat the first point at the end.
# These are illustrative anatomical vertices, not universal shirt coordinates.
target = np.array([[60,155], [60,70], [30,85], [20,60], [65,30], [90,35]])
#             hem       armpit    cuff       cuff     shoulder    neck
outline = OuterContourDetector().detect(boundary_result.object_mask)

# Only vertices 2 and 3 move. Torso/shoulder/neck vertices remain fixed.
sleeve = OutlineArticulation(
    start_index=2, end_index=3, pivot_xy=(62.5,50),
    max_angle_deg=30, angle_step_deg=10,
)
matcher = PartialOutlineMatcher(fraction_range=(.20,.55))
result = outline.match_partial(target, matcher=matcher, articulation=sleeve)
print(result.status, result.matched)
if result.best is not None:
    print(result.best['start_xy'], result.best['end_xy'])
    print(result.best['metrics'], result.best['reasons'])
    print(result.best['scale'], result.best['articulation_angle_deg'])
image_rgb = result.render_debug(color_rgb)
result.save_debug('outline-match.png', color_rgb)  # explicit write; no overwrite
```

`matcher.match(target, outline, articulation=...)` is equivalent. Pass the correct
`observed_mask` to `OuterContourDetector` when the image is clipped or unobserved.
Its frame/observation-boundary edges cannot confirm a match. If segmentation is
uncertain, fix or inspect it first: the matcher cannot discover segmentation errors
that were declared observed. Do not reuse internal `measure_components` trails as
exteriors; their branches/gaps have a different meaning.

For an independently extracted simple closed contour, use
`matcher.match_contour(target, contour_points_xy, observed_edges=bool_array,
articulation=sleeve)`. Edge `i` runs from vertex `i` to `(i+1) % N`.
Omit `observed_edges` only when every edge is observed. The last point must not
duplicate the first. Both APIs copy inputs and never write during matching.

## Existing classes: runnable integration examples

Run `PYTHONPATH=src python examples/example_outline_match_sources.py` to execute
21 routes on one synthetic shirt scene. Add
`--output-dir /existing/parent/new-folder` for a JSON report and numbered PNGs.
The example refuses an existing output directory. The source contains a separate
commented block for each class, including actual detector/segmenter calls; no
result objects are fabricated. The matching library itself is unchanged.

Use these output fields with the example's `match_mask(...)` helper, which calls
`OuterContourDetector.detect(...)` followed by `match_partial(...)`:

| Producer / result | Input to matching | Observation / scope |
|---|---|---|
| Caller bool mask | Full `object_mask` | Retain the caller's silhouette observation domain. |
| `OuterContourDetector` / `OuterContourResult` | Prefer its own `match_partial(...)`; optionally `component_mask(contour_id)` | Existing `observed_mask`; component helper preserves holes and selects one component. |
| `OverheadBoundaryDetector` / `BoundaryResult` | `object_mask` | Retain the actual silhouette observation domain and inspect segmentation stages. `uncertain_mask` is not a universal observation mask; a rejected background can mark the whole domain uncertain. |
| `RGBDBackgroundSegmenter` / `RGBDBackgroundResult` | `object_mask` | Use `observed_mask` for the public union/depth-required modes. If an extended implementation provides **`outline_observed_mask`**, prefer it to retain suppressed/deferred gaps; the example handles both APIs. Do not erase all uncertain pixels. |
| `OverheadCornerDetector` / `CornerResult` | `object_mask` | Retain the original silhouette observation domain; depth support for corner classification is separate. |
| `SeededColorAssistant` / `ColorAssistResult` | `object_mask` | Original observation AND `~uncertain_mask` AND `~hand_exclusion_mask`. The example has two usable appearance classes and uses `boundary_guard_px=0`; default erosion introduces an uncertain removed band. Neither setting proves semantic identity. |
| `ReferenceGripperSegmenter` / `ReferenceSegmentationResult` | `object_mask` | `observed_mask` minus uncertainty, robot and hand masks. Keep contact/blocked-region cuts unknown. Requires the declared fixed-view, repeatable-pose reference. |
| `BoundaryResult.filter_outer_proximity()` / `BoundaryProximityResult` | `object_mask` | Reuse its `observed_mask`; graph selection does not produce a new exterior. |
| `DepthSurfaceDetector` / `SurfaceResult` | `object_mask` | Retain original silhouette observation and exclusions. Depth-fit `unknown_mask` is not silhouette availability. |
| `OuterEdgeBulgeDetector` / `EdgeBulgeResult` | `object_mask` | Retain original silhouette observation. Relief candidates/reference/unknown masks do not replace the silhouette. |
| `OverheadPickCandidateDetector` / `PickCandidateResult` | `object_mask` | Selected component only; reuse the originating outline's `observed_mask`. Pick-search exclusions are not exterior observations. |
| `BridgeDrapeDetector` / `BridgeDrapeResult` | `foreground_mask`, conditionally | Only when this foreground is the intended object, as with the demonstrated `detect_mask` route. Keep `~invalid_mask` and actual occlusion/exclusion information. Difference-based foreground may include hands/clutter. |
| `analyze_color_geometry(...)` | Returned `BridgeDrapeResult.foreground_mask` | Preserve the color classification/exclusion domain and `~invalid_mask`. Single-class appearance candidates do not establish object identity. |
| `LiftDetector` / `LiftResult` | `diagnostics['object_mask']`, only with `diagnostics=True` | Original input mask, not a new segmentation; retain the original observation/exclusion domain. `valid_mask` is object-restricted depth support and cannot serve as a whole-image observed domain. |
| `IndependentGraspDetector` / `GraspResult` | `diagnostics['eligible_object_mask']`, only with `diagnostics=True` | Retain `~caller_unobserved_mask & ~ambiguous_label_mask`; the excluded label boundaries must stay unknown. This mask is caller-derived, not inferred from grasp success. |

`ObjectOnlyGraspDetector` consumes an object mask but its result does not preserve
a reusable full object mask. Its example block retains the **original input** and
passes that to matching alongside the grasp result. Likewise, if diagnostics were
disabled for Lift/Grasp, keep the input mask instead of reconstructing it from
measurements. Apply known exclusion masks to the observation domain so artificial
cuts cannot be confirmed as physical edges. Do not substitute `object_mask` itself
for `observed_mask`: the object border then becomes an observation-domain border.

Three stored point-list routes are also demonstrated with `match_contour(...)`:

| Stored ordered closed exterior | Conditions |
|---|---|
| `BoundaryResult.exterior[i]` | Simplified external contour. Long edges need full-edge observation checks, not just endpoint checks. Does not carry its own visibility metadata. |
| `CornerResult.diagnostics['contours_xy'][i]` | Requires `diagnostics=True`; source-area filtering can omit small components. These are contours, unlike `candidates`. Retain the original observation domain. |
| `OuterContourResult.contours[i]['points_xy']` | Keep the source status (`computed`), observation and contour identity. The normal `outline.match_partial(...)` API preserves these automatically and is preferred. |

The helper `observed_exterior_edges` checks every rasterized edge against the
declared observation domain with the default outline frame guard. It handles
integer-pixel Boundary/Corner contours, including image-edge contact and missing
pixels between distant stored vertices. It does not infer visibility from color
or fill gaps. For independent subpixel point lists, provide their own per-edge
`observed_edges` to `match_contour`; do not round them through this helper.
All current point lists must be closed exteriors in cyclic order, without repeating
the first point. An open current fragment is not supported by this API.

**Separate per-contour calls do not test ambiguity between objects.** The example
retains a source index and reports each result separately. Two `matched` records
are not a unique location. Use `OuterContourResult.match_partial` to search all
components jointly and report competing locations as `ambiguous`.

Not suitable as full current-object exteriors:

- `BoundaryResult.lines`, color/depth boundary masks, topology segments,
  `measure_components(...).trail` and proximity kept/excluded graph paths: internal
  image evidence, potentially branching/open/inferred; not an ordered object loop.
- `OuterContourResult.holes`, individual open `segments`, `CornerResult.candidates`,
  and `PartialOutlineMatchResult.best['points_xy']`: holes, isolated corners or open
  arcs, not the full current exterior. A trusted open arc may be a **target** only
  after checking its meaning and order; do not automatically turn a predicted match
  into a new reference.
- Pick/contact/elevated/nonflat/reference/near masks and `BridgeDrapeResult.work_mask`
  or inferred bridge corridors: selected measurement subsets or artificial cuts.
- `GripperReference`, `PlaneFitResult`, `CameraIntrinsics`, `TablePlane`: reference
  or calibration data, not a current object silhouette. `ObliqueShapeAnalyzer` is
  internal analysis whose paths/feature masks are not a separate full-object source.

These examples check API composition and visibility handling on synthetic data;
they do not establish segmentation correctness or real-shirt matching accuracy.

## Algorithm and tradeoffs

1. A bounded pixel simplification (default 0.75 px distance-to-segment tolerance)
   removes raster staircases before measuring arclength. Without it, rotating a
   pixel mask can change arc parameterization despite unchanged object geometry.
   Ramer–Douglas–Peucker preserves original vertex order; a simplification that
   introduces a crossing is discarded. Set `simplify_px=0` to disable. This is a
   resolution-scale noise allowance: detail below it is not a reliable fold signal.
   Uniform arclength samples then represent each template. An optional, **explicitly
   declared** contiguous vertex group is rotated around a fixed pivot to form a
   finite family of acceptable silhouette hypotheses. The surrounding fixed
   vertices reconnect by line segments. Crossing/degenerate variants are omitted.
   Without an articulation declaration, only the original target is used.
2. Enumerate start positions and arc lengths on the cyclic current contour, in
   both traversal directions, including intervals across the array seam. Each
   candidate must cover the entire target. Missing target sections are not skipped,
   separate fragments are not joined, and hole contours are excluded.
3. Fit a least-squares, orientation-preserving 2D similarity transform to each
   correspondence: translation, rotation and a single uniform scale. Reflections,
   anisotropic scale, perspective warps, DTW and arbitrary elastic deformations
   are excluded. Correspondence is normalized arclength, not nearest-point hopping.
4. Score six independent guards: RMS and worst-point position error (divided by
   current arc length); signed local turn RMS at two support radii; excess absolute
   fine turn; and excess arclength lost during resampling. The latter guards help
   reject wrinkles/backtracking that a global fit or coarse sampling could hide.
   Turning uses chords at radii 2 and 5 of 64 samples by default, avoiding raw
   one-pixel staircase angles. No free deformation is optimized after alignment.
5. Refine up to 12 distinct spatial seeds three times in start/length space.
   Suppress alternatives whose arc overlap exceeds 60% of the shorter arc, then
   compare distinct passing locations. All traversed raw edges must be observed,
   including short unavailable edges between resampled points. A simplified edge
   is marked unavailable if **any** raw edge it spans was unavailable; simplifying
   cannot hide missing observations. This conservative propagation may withhold
   a larger interval than the original unavailable edge.

This is a bounded heuristic, not globally optimal partial elastic registration.
Finite sampling can miss the correct interval; changing search density may change
the result. `diagnostics.search_complete` is always false. Near-threshold cases
need a denser search and visual review. Candidate suppression can merge nearby
valid placements, so absence of an ambiguity flag does not prove uniqueness.

Global Hu-moment comparison does not provide an ordered open-to-subarc alignment
([OpenCV shape API](https://docs.opencv.org/4.0.1/d3/dc0/group__imgproc__shape.html)).
Shape contexts with flexible correspondence/deformation are a useful broader
alternative ([Belongie, Malik and Puzicha](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/pami.html)).
Here we deliberately restrict the deformation family so folds cannot be explained
away by unconstrained warping. These are design references, not imported algorithms
or claims that the matcher reproduces their results.

## Sleeve-angle configuration

The matcher cannot infer which points belong to a sleeve or where the shoulder
pivot is. Mark these on the reference. For a contour sampled at every pixel,
first create an ordered anatomical polyline; rotating an arbitrary index range
can create unrealistic connecting edges. One movable range is supported; there
is no independent rotation at every corner. A range can touch either open endpoint
but cannot contain the entire target. Indices are inclusive and refer to the
original target, before resampling. Reverse the index mapping if reversing the
target; reversing the current contour requires no configuration change.

`max_angle_deg` is 0..45, default 30; `angle_step_deg` is 1..45, default 10.
The evenly spaced bank includes zero and both extrema, with spacing at most the
requested step. The reported angle selects a template hypothesis; it is not a
measured physical joint angle. Position/turn tolerances can admit additional
difference beyond the bank angle, so the bank is not a hard bound on the actual
sleeve rotation. Keep tolerances conservative and include near-limit negatives
when tuning. Automatic hinge discovery, multiple articulated parts, fabric stretch,
perspective and hidden-edge completion are outside this initial implementation.

## Decisions and diagnostic fields

| `status` | `matched` | Meaning |
|---|---|---|
| `matched` | `True` | A fully observed candidate passes all configured guards; no retained competitor is close enough to flag ambiguity. |
| `not_matched` | `False` | No candidate passes in the sampled search of the supplied complete contours and allowed fraction range. Not proof that no correspondence exists. |
| `unknown` | `None` | Missing/uncertain evidence, empty/tiny/ambiguous topology, or insufficiently informative near-straight target. |
| `ambiguous` | `None` | Distinct visible locations have similar passing scores; existence evidence is available but location is not unique. |

An unrelated missing region does not erase a supported positive match. If no
supported match exists and any contour evidence is incomplete, the result is
unknown, not a negative claim about a hidden part. Shape alone cannot identify
the garment, distinguish identical objects, detect all overlap, or determine
physical folds that leave the same silhouette. A nearly straight target is
uninformative after free scaling and returns unknown.

`best` can also be an unsuccessful or unsupported diagnostic candidate. Always
check `status`/`matched` before using it as a match. `candidates` contains up to
five distinct retained locations by default. Each candidate includes:

- `contour_id`, `start_xy`, `end_xy`, `points_xy`: current-image coordinates in
  target traversal order; endpoints may be interpolated fractional pixels.
- `fraction`, `arc_length_px`, `start_arc_px`, `end_arc_px`, `reversed`: interval
  position in the simplified cyclic current contour, anchored at raw vertex zero.
  Input and matching perimeter/vertex counts appear in `diagnostics.source_contours`.
  Start/end coordinates interpolate the simplified segments and can differ from
  raw pixel-center edges by the simplification tolerance. Reverse traversal is allowed,
  but mirror reflection is not. Fractions refer to the **current** perimeter.
- `fitted_target_xy`: transformed and sampled selected template; `scale`,
  `rotation_deg`, `translation_xy`, `articulation_angle_deg` describe the transform.
  Apply articulation first, then `scale * rotation @ point + translation`.
- `metrics`, `score`, `supported`, `passed`, `reasons`: `score` is the largest
  metric-to-limit ratio. All six guards must be <=1. It is not a probability or
  calibrated confidence. `supported` additionally requires observed raw edges and
  minimum current arc length. A passing candidate is not necessarily unique.

`to_dict()` returns detached strict-JSON-compatible data, including parameters,
source contour reasons, template angles and evaluation count. Source pixels are
not included. `render_debug(color, color_order='RGB', candidate_index=0)` returns
a new uint8 image: cyan current samples, orange fitted template, white endpoints.
The input image must use the same coordinate frame. `save_debug` has the existing
PNG/existing-parent/no-overwrite safeguards. No background RGB is required to
perform matching.

## Defaults and tuning

| Parameter | Default | Effect |
|---|---|---|
| `fraction_range` | `(0.20, 0.55)` | Search arc lengths as fractions of current perimeter. Use a broad prior because folds can change that perimeter. Increasing this range increases false alternatives and may need more length samples. |
| `search_samples` | 128 | Coarse cyclic start positions, integer 32..512. |
| `length_samples` | 25 | Coarse lengths across the fraction range, integer 3..101. |
| `sample_count` | 64 | Uniform points per arc, integer 32..128. Changing it also changes turn support and requires retuning. |
| `simplify_px` | 0.75 | Distance-to-segment tolerance before arclength measurement, processing pixels 0..5. Removes staircase/noise; can remove real details at that scale. Use 0 for unsimplified geometry. Applied separately in target and current coordinate units. |
| `max_rms` | 0.020 | Similarity-fit RMS / current arc length. |
| `max_point_error` | 0.055 | Worst correspondence distance / current arc length. |
| `max_fine_turn_deg` | 20 | RMS wrapped signed-turn difference, radius 2 samples. |
| `max_coarse_turn_deg` | 25 | Same, radius 5 samples. |
| `max_excess_turn_deg` | 60 | Positive excess sum of absolute radius-2 turn / 2. Approximate roughness guard, not exact integrated curvature. |
| `max_unresolved_arc_ratio` | 0.06 | Positive excess of `1 - sampled_chord_length/simplified_arc_length` compared with template. Helps expose detail lost between samples after pixel simplification. |
| `ambiguity_margin` | 0.12 | Flag another distinct passing location within this score difference of the best. |
| `min_arc_length_px` | 30 | Minimum observed support in current-image pixels. |
| `max_candidates` | 5 | Retained distinct diagnostic locations, integer 2..20. |

Defaults are provisional synthetic settings. Increasing geometric/turn limits
accepts more shape variation **and more folds**. Calibrate with independently
labeled good and bad garments at the actual camera scale, including near-limit
angles, occlusion, seam/segmentation noise and multiple objects. Use held-out
garments/scenes to choose thresholds; do not tune only on the examples below.
Uniform scale is unconstrained, so a small lookalike can fit if it meets the pixel
support limit. Physical dimensions need separate calibrated evidence.

Raw current contours above 4096 vertices return unknown because their independent
topology check is quadratic; `OuterContourResult` reuses extractor topology instead.
Targets allow at most 2048 vertices, and individual numeric inputs at most 20000.
The existing outline extractor's computation/ambiguity statuses remain unknown.
There is no segmentation, semantic model or additional package dependency; raw
point matching uses NumPy, while masks/debug rendering use optional OpenCV.

## Validation scope

Run `PYTHONPATH=src python -m unittest tests.test_outline_match -v` and
`PYTHONPATH=src python examples/example_outline_match.py`.
The example accepts `--output-dir /existing/parent/new-folder` and saves JSON,
individual overlays and `comparison.png`. It refuses an existing directory.

Tests cover similarity/scale/reversal/seam invariance, bounded off-grid sleeve
angles on raster masks, folds and unrelated silhouettes, high-frequency contour
disturbance, small coordinate noise, short missing edges, independent observed
arcs, frame/domain clipping, ambiguity between objects, hole exclusion, invalid
inputs, deterministic results and strict JSON. The synthetic controls do not
establish real-shirt classification accuracy. Existing real masks can exercise
integration and self-retrieval, but no verified good/bad partial-shirt matching
labels or independent target/current pairs were available for this task.
