# Interior-corner alignment and separate edge checks

`CornerOutlineMatcher` is an **opt-in alternative** to `PartialOutlineMatcher`.
It fits the internal bends of an open target to ordered bends on a closed
current exterior, excluding both target endpoints from the fit. It reports
global rotation, separate target-axis scales, explicit sleeve rotation, corner
residuals, and errors for every edge, including the two endpoint tails.
The legacy matcher and its defaults remain unchanged.

```python
from depth_image_lens import CornerOutlineMatcher, OutlineArticulation

matcher = CornerOutlineMatcher(
    axis_angle_deg=0,              # X/Y axes in the supplied target frame
    scale_range=(0.5, 2.0),        # applies to EACH axial scale
    axis_ratio_range=(0.75, 4/3),  # scale_x / scale_y, independent of size
    max_rotation_deg=180,
    fraction_range=(0.15, 0.65),  # current RAW perimeter fraction
    max_skipped_corners=0,       # opt in with 1 or 2 for extra CURRENT bends
)
result = current_outline.match_partial(target_xy, matcher=matcher,
                                       articulation=None)
# Equivalent: matcher.match(target_xy, current_outline)
# Or: matcher.match_contour(target_xy, current_closed_xy,
#                           observed_edges=current_edge_observed)
print(result.status, result.matched)
if result.best is not None:
    b = result.best
    print(b['rotation_deg'], b['scale_x'], b['scale_y'], b['axis_ratio'])
    print(b['scale'])  # geometric mean sqrt(scale_x * scale_y), not an independent fit
    print(b['articulation_angle_deg'])
    print(b['score'], b['geometry_score'], b['limit_ratios'], b['reasons'])
    for edge in b['edges']:
        print(edge['index'], edge['role'], edge['length_ratio'], edge['rms_px'],
              edge['metrics'], edge['score'], edge['passed'])
debug_rgb = result.render_debug(current_rgb)
```

Run `PYTHONPATH=src python examples/example_corner_outline_match.py` for a complete
synthetic comparison and a dense same-mask reference example. Optional
`--output-dir /existing/parent/new-folder` saves JSON and debug PNGs without
overwriting an existing directory. Cyan is the observed arc; orange is the fitted
target, both in the current image frame. Endpoints are white.

## Targets and corner correspondence

Targets are ordered open XY polylines of 3–2048 vertices without duplicate
neighbors, a repeated closing point, or self-intersection. Current contours are
ordered closed exteriors without a repeated closing point. Both traversal
directions and the array seam are searched. Mask and existing-class routes in
[OUTLINE_MATCH.md](OUTLINE_MATCH.md) accept this matcher too; their observation
and topology requirements still apply. Keep original curves in the target:
corner-only chords cannot describe a curved edge between those corners.

The matcher finds bends independently of `OuterContourResult.corners`:

1. RDP corner simplification, `corner_tolerance_px=1.5` in each input's pixels.
2. Absolute turn at least `corner_turn_deg=20` degrees.
3. Strongest-bend suppression within `corner_spacing_px=5` arclength pixels,
   to avoid treating adjacent raster steps as separate corners.

Target endpoints are excluded. At least **three non-collinear interior bends**
must remain. Two points do not identify translation, rotation, and two scales.
`min_corner_condition=.08` is the smaller/larger singular-value ratio of the
centered target corner coordinates. Nearly collinear targets return `unknown`.
The maximum target corner count is 16; current count is 64 per traversal.

By default each correspondence uses consecutive detected current corners.
`max_skipped_corners=1` or `2` permits a limited number of extra **current**
corners while retaining order. It does not skip a target corner whose counterpart
is missing, bridge an unobserved gap, or infer semantic sleeve corners. A real
bend that becomes too shallow to detect can therefore still cause a false
negative or `unknown`. Small raster/scale changes can change detection. These
are detection limits, not evidence that cloth folded.

Endpoints are projected continuously onto the current contour **before the first
and after the last matched corner**, bounded by the neighboring detected corners
(or the extended windows allowed by the skip budget below).
They are not restricted to corner points or a finite start-position grid. The
two tails are still checked, so excluding endpoints from the fit does not let a
bad endpoint or tail disappear. This is a different search from the legacy
whole-perimeter start/length grid; it cannot recover a correspondence whose
internal corners were not detected. `fraction_range` uses the **raw** current
perimeter; the legacy matcher uses its simplified perimeter.

`target_corner_indices` indexes the original input target. `observed_corner_indices`
indexes the original current contour even for reversed matches.
`target_corner_xy` contains articulated target coordinates, `fitted_corner_xy`
contains their transformed coordinates, and `observed_corner_xy` contains the
corresponding current corners. Do not confuse these with detector corner IDs.

## Limited extra-current-corner skips

```python
matcher = CornerOutlineMatcher(max_skipped_corners=1)
result = current_outline.match_partial(target_xy, matcher=matcher,
                                       articulation=sleeve_or_none)
if result.best is not None:
    print(result.best['skipped_corner_count'])
    print(result.best['skipped_corner_indices'])  # original CURRENT contour indices
    print(result.best['skipped_corner_xy'])       # current-image coordinates
    print([edge['score'] for edge in result.best['edges']])
```

Start with 1; use 2 only if necessary. The default 0 retains the preceding
algorithm's decisions and scores. The hard maximum is 2. The bound is the
**total for the whole matched arc**, not a separate allowance for each edge.
It includes extra corners between matched internal corners and on either
endpoint tail. Endpoints themselves are not counted as skipped corners.

Skipping removes a corner only from transform correspondences. The original
intervening polyline still supplies edge samples, length and turn errors, and
every raw edge still supplies its observation flag. Thus a small spurious bend
may pass while a large fold at the same place fails its edge guards. Existing
`simplify_px` and finite resampling limits still apply: tiny geometry can be
smoothed and there is no guarantee that every physical fold is rejected.
Skipped corners are not automatically labeled noise; these are just alternative
ordered correspondences. No target corner can be skipped. A target with an extra
corner absent in the current image remains a mismatch/uncertain case.

There is no extra skip penalty in `score`; geometry and deformation guards
remain directly comparable. Exact score ties prefer fewer skipped corners.
All potentially passing hypotheses remain eligible for placement/transform
ambiguity, even after one passing candidate has been found. A cheap corner-fit/
deformation lower bound prunes hypotheses only if they cannot pass or improve
the best retained failure. It does not discard a candidate merely because its
corner fit is worse than an existing passing candidate.

For M target internal corners, S permitted skips and sufficient current corners,
there are at most `binomial(M+1+S, S)` correspondence/endpoint-window choices per
current starting corner and articulation angle. Both traversal directions are
searched. For M=4 this is 1 / 6 / 21 choices for S=0 / 1 / 2. Tail windows reuse
the interior transform fit, and the lower bound avoids many expensive edge
comparisons. This bounds growth, but does not make the extra search free.

`max_hypotheses` (default/hard maximum 4096) counts the full correspondence ×
angle × endpoint-window hypotheses, including cheap rejections. A conservative
preflight check refuses a direction if its planned search would exceed the
remaining budget; it does not silently reduce the skip setting. Exhaustion gives
`unknown` / `search_budget_exceeded`, even if a partially searched candidate
passed. `diagnostics.evaluated_hypotheses`, `corner_bound_pruned_hypotheses` and
`endpoint_window_evaluations` describe work performed; the last counts calls to
endpoint projection/candidate evaluation, including fraction-range rejections.
`max_candidates` limits returned rows, not computation.

Run `PYTHONPATH=src python examples/example_corner_outline_skips.py --repeats 3`
for a same-input 0/1/2 comparison. It checks clean input, one/two extra corners,
two endpoint tails, large folds, missing corners, shear, a 48-corner repeated
shape, and budget exhaustion. Add `--output-dir /existing/parent/new-folder`
for JSON and diagnostic PNGs. These are synthetic controls and timing samples,
not real-image noise/fold accuracy.

Skipping is not sufficient when the edge itself still fails: a low-resolution
raster version of the small-bend fixture scored 1.082 after its extra corner was
skipped and remained `not_matched`; at twice the raster resolution it scored
0.984 and passed. Keep the edge diagnostics when tuning, rather than assuming
every skipped corner is harmless or increasing its allowance indefinitely.

## Transform meaning and sleeve configuration

The transform has no shear or reflection. For column coordinates:

`current = R(global_angle) A diag(scale_x, scale_y) A.T articulated_target + translation`

Here `A=R(axis_angle_deg)` defines fixed axes **in the input target frame**. With
zero axis angle they are its X/Y image axes. They are not automatically garment
width/height axes. If the reference shirt is tilted, declare its axes, or rotate
the target and pivot consistently before use. Moving the target origin changes
translation only; rotating the axes can change the allowed shape family.
Positive angles use image XY coordinates (visually clockwise when Y points down).
`matrix_xy` is the complete 2x2 matrix after articulation, with `translation_xy`.

The analytic least-squares fit uses only interior corner correspondences. It fits
independent positive scales and an orthogonal rotation, rejecting collapsed or
reflected solutions. It does not fit a free affine matrix and reinterpret shear
as scale. Thresholds validate the fitted parameters; a fit outside their bounds
is rejected rather than clipped to the bound and reoptimized.

Use the existing explicit articulation declaration:

```python
sleeve = OutlineArticulation(
    start_index=2, end_index=3, pivot_xy=(62.5, 50),
    max_angle_deg=30, angle_step_deg=10,
)
result = current_outline.match_partial(target_xy, matcher=matcher,
                                       articulation=sleeve)
```

Those illustrative indices refer to the six-point anatomical target in the
example. Reassign them for a dense target. The range is inclusive and the pivot
is in that target's coordinate frame. One region rotates before axial scaling;
adjacent fixed vertices reconnect by straight segments. No local length-stretch
parameter exists. Angles are enumerated, including zero, within the declared
range (maximum configurable range ±45 degrees). Every surviving angle's corner
correspondences are evaluated; no coarse spatial seed can displace the rigid
angle as in the legacy search regression. No corner indices are re-detected on
each rotated target hypothesis. Strong distortion can thus invalidate a
correspondence instead of being absorbed as a different corner topology.

## Scores, acceptance and uncertainty

The overall `score` is the maximum of independent limit ratios; **smaller is
better, at most 1 passes the numeric guards**. This is not a probability. Unlike
the legacy score, it includes deformation amounts, so scores between matchers
are not numerically interchangeable. A perfect geometric fit with scale 0.8 has
a nonzero overall score. `geometry_score` isolates corner and edge agreement.

| Guard | Default / normalization |
|---|---|
| Corner RMS / maximum error | `.025` / `.06`, divided by fitted corner bounding-box diagonal |
| Each axial scale | `(0.5, 2.0)`; absolute log displacement from 1 divided by the relevant bound's log displacement |
| Axial scale ratio X/Y | `(0.75, 4/3)`; same log normalization |
| Global rotation | `180` degrees; absolute angle / limit |
| Sleeve rotation | Absolute angle / declared maximum; zero when disabled |
| Each edge RMS / maximum error | `.06` / `.15`, divided by that fitted target edge's arclength |
| Each edge length error | `.20`, absolute value of observed/fitted-target length ratio minus 1 |
| Each edge turn RMS / excess turn | `25` / `60` degrees |
| Each edge unresolved arclength | `.06` excess loss relative to the target during resampling |

Each edge (including tails) is simplified with `simplify_px=.75` in current
pixels and compared with `edge_samples=32` uniform arclength samples. Corners
anchor correspondence, so length changes in one interval do not shift every
other interval. This is not unrestricted DTW: each interval has its own length,
position and shape guards. Raster details smaller than the configured
simplification/spacing are not reliable fold evidence. Edge scores can vary with
short-edge rasterization and direction; exact zero is not promised for a mask.

- `matched`: a fully supported candidate passes all deformation and geometry
  guards and no tested comparable ambiguity is found.
- `not_matched`: no tested correspondence passes on sufficient observed input.
  Corner-detection mismatch is a possible cause, not necessarily a fold.
- `unknown`: too few/collinear target corners, insufficient current corners,
  invalid topology/status, unavailable evidence, empty input, or a computation
  budget prevents a decision.
- `ambiguous`: distinct visible placements fit, or materially different global
  rotations, scales, or sleeve angles explain the same placement comparably.
  `diagnostics.alternative_transforms` exposes competing deformation estimates.

Ambiguity uses `geometry_score` within `ambiguity_margin=.12`, not the deformation
prior. A preference for less deformation does not identify the physical cause.
Material differences default to `ambiguity_rotation_deg=15` for global or sleeve
rotation, or `ambiguity_scale_ratio=1.15` for either axial scale. Only passing
hypotheses compete. Nearby placements with at least 60% arc overlap count as the
same location. Ambiguity detection is finite and does not prove uniqueness.

Every raw traversed edge must be observed, including gaps shorter than the sample
spacing; `min_arc_length_px=30` also applies. Missing edges on the opposite side
do not invalidate a supported local match. `max_hypotheses=4096` bounds total
correspondence/angle/endpoint-window evaluations and budget exhaustion returns `unknown`, even
if an earlier candidate passed. Current input is limited to 4096 vertices per
contour in this mode. `diagnostics.search_complete` remains false.

## Evidence and tradeoffs

Tests cover global pose, axial scale, declared axes/origin, sleeve rotation and
its combination with scale/pose, projected non-corner endpoints, dense mask
references, folds, extra/missing corners, shear, reflection, sub-sample observation
gaps, competing objects and global/local rotation confounding. The runnable
example compares both matchers on the same synthetic continuous polygons and
reports separate runtimes and outputs. A shear example passes the legacy shape
tolerances but fails the new per-edge checks. This demonstrates a differing
constraint, not overall superiority or measured real-shirt accuracy.

A separate smoke check used four existing provisional real-image masks, each
with its own fixed 10%-start, 35%-vertex-count target window. This matcher returned
one `matched`, one `unknown` for nearly collinear internal corners and two
`unknown` for only two internal corners; the legacy matcher matched all four.
This is same-contour retrieval, not an independently labeled good/folded test.
It reinforces that some useful real outlines cannot identify two axial scales
from their corners. No mask/RGB data is bundled with these synthetic examples.

Use the legacy matcher for shapes with too few stable corners or when approximate
arclength matching is preferred. Use this matcher when the internal bends are
reliably visible and interpretable deformation/edge diagnostics are important.
Actual shirt images, segmentation noise, camera projection and annotated good/
folded examples still need calibration. Anisotropic scale is an image-coordinate
description, not proof of cloth stretching, physical dimensions, or perspective
correction. Neither matcher certifies the condition of the unexamined garment.
