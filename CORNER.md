# Visible overhead corner candidates

`OverheadCornerDetector` is an independent experimental feature. It finds convex,
near-right-angle **visible outer-contour** candidates and reports local depth
anomalies. It does not establish single-layer geometry, physical right angles,
material identity, or safe grasping. It does not replace BridgeDrape or Lift.

## Install and inputs

The base package remains NumPy-only. This optional feature uses OpenCV contours,
polygon approximation and distance transforms. Use an existing compatible OpenCV
installation, or install the `corner` extra in your virtual environment:

```bash
python -m pip install -e '.[corner]'
python examples/example_corner.py
python -m unittest tests.test_corner -v
```

No model, training or network access is used at detection time. The development
environment already had OpenCV 4.6.0; no dependency download was needed for this pass.
OpenCV >=4.5.5 uses Apache-2.0; NumPy retains BSD-3-Clause. Neither is vendored.
Original implementation code remains MIT. See [OpenCV licensing](https://opencv.org/license/).

```python
from depth_image_lens import OverheadCornerDetector

result = OverheadCornerDetector().detect(
    color, depth,                         # uint8 HxWx3; numeric HxW optical-axis Z
    color_order="RGB",                   # or "BGR", never guessed
    registered_to_depth=True,            # caller assertion, not an alignment test
    object_mask=object_mask,              # bool HxW in the same pixel frame
    depth_scale_m=0.001,                  # mm/count; use 1.0 for metres
    valid_mask=original_valid_mask,       # optional bool HxW
    diagnostics=True,
)
print(result.to_dict())                   # strict-JSON-safe; excludes arrays
```

Supply exactly one of `object_mask` or `background_color`. The latter must be a
matched uint8 empty-scene color image of the same shape/order. Its Euclidean
color difference is a simple mask baseline, not semantic segmentation. A supplied
mask gives a controlled geometry test; it is not evidence that RGB extraction works.
Patterns can be handled when the outer mask is correct, but similar background
colors, illumination changes and occluding arms can break the mask baseline.
No hand/object separation is performed here.

Depth zero, negative, NaN and infinity are invalid; false `valid_mask` remains
invalid. Missing depths are not filled. A positive sensor sentinel must be marked
invalid by the caller. Scaling is explicit for **every dtype**, including uint16
and float32. Inputs are not modified. A background-difference visualization is
not raw depth and must not be substituted. Radial range must be converted to
optical-axis Z upstream. Color/depth same dimensions do not prove registration;
rectification, crop, resize and registration must already be consistent. The
boolean declaration does not estimate or repair misalignment.

Optional `intrinsics=CameraIntrinsics(...)` changes local interior residuals to
camera-frame 3D plane distances. Otherwise an affine optical-Z trend over pixels
is used; this is approximate under perspective. Internal step and boundary tests
still use optical-Z residuals. No world/robot calibration or global table plane
is required. Intrinsics must be updated after crop/resize. Angles remain 2D in
both modes; a perspective view can distort a physical right angle.

## Algorithm and output

1. Extract external contours and approximate polygons. No convex hull creates
   missing corners, and texture corners inside the mask are not proposals.
2. Keep convex near-right angles with adequate edge extent. Fit local lines on
   each side at two window lengths; check residual, intersection displacement
   and angle stability. Cropped boundary candidates remain unknown.
3. In `mode="depth"` (default), inspect original valid interior points of the
   candidate's own 8-connected component, preserving mask holes and excluding
   a two-pixel boundary band. Check local trend/plane residual and internal depth
   steps. An ordinary object/table step outside the mask is reported separately
   and is not itself fold evidence. Boundary inconsistency causes unknown.

`mode="rgb"` is the polygon-angle baseline; `mode="refined"` adds line fitting.
These modes still require the same input contract but do not check depth quality.

Each candidate has `xy`, `polygon_xy`, `angle_deg`, `status`, `reasons`, and
`evidence` (line residual/stability, original depth validity, interior residual,
internal step count, outside-minus-inside Z and boundary error when available).
All stages report rejected proposals too. `accepted` means a visible candidate
passed the selected checks; `suspected` means a geometry/depth check failed,
not proven overlap; `unknown` means insufficient or inconsistent evidence.
`single_layer_status` is **always unknown**. Summary counts and
`validation_status="not_validated"` keep measured output separate from validation.
No heuristic probability is returned. `diagnostics` contains contour coordinate
arrays, component labels, the interior mask and original valid-depth mask;
`object_mask` is separate. Nearby separate-component occupancy and valid-depth
pixel counts are recorded separately, not mixed into the target interior fit.
Outside depth statistics exclude all components. Connectivity is a mask property,
not semantic identity: touching components cannot be separated by this rule.

A perfectly aligned hidden layer can produce exactly the same RGB/depth inputs
as one layer. Four such corners pass every stage in the regression set. Neither
height, local planarity nor this detector can resolve that counterexample.
Registration errors may yield suspected internal steps or unknown boundaries;
some offsets remain undetected. Local checks do not inspect the entire object.

## Parameters and tuning

First verify units/registration and the mask, then inspect diagnostics. Defaults
are illustrative development settings, not validated sensor thresholds.

| Parameter | Default | Purpose |
|---|---:|---|
| `min_area_px` | 100 | Minimum contour area |
| `polygon_epsilon_px` | 2 | Polygon approximation tolerance |
| `edge_window_px` | 18 | Local contour arc length for each fitted side |
| `angle_tolerance_deg` | 25 | Legacy fallback for both angle tolerances below |
| `proposal_angle_tolerance_deg` | `None` | Polygon deviation from 90 degrees; `None` inherits `angle_tolerance_deg` |
| `fitted_angle_tolerance_deg` | `None` | Fitted deviation from 90 degrees; `None` inherits `angle_tolerance_deg` |
| `angle_stability_mode` | `"reject"` | `"diagnostic"` records/warns without rejection by angle difference alone |
| `angle_stability_deg` | 10 | Maximum long/short fitted-angle difference, also recorded in diagnostic mode |
| `max_intersection_shift_px` | 2 | Maximum long/short fitted-intersection displacement |
| `min_fit_support` | 4 | Minimum original contour samples on each side in **both** windows (integer >=3) |
| `line_residual_px` | 1 | Maximum local line RMS residual |
| `depth_radius_px` | 24 | Local depth inspection radius |
| `depth_step_m` | .015 | Internal adjacent-pair step threshold, after local Z trend subtraction |
| `boundary_mixing_m` | `None` | Boundary-trend error p95; `None` inherits `depth_step_m` for compatibility |
| `depth_residual_m` | .005 | Interior residual p95 threshold |
| `color_distance` | 35 | Optional RGB/BGR background difference threshold |
| `support_policy` | `"fixed"` | Opt-in `"adaptive"` bounded short-window support |
| `proposal_policy` | `"single"` | Opt-in `"multiscale"` bounded polygon union |
| `proposal_merge_px` | 3 | Component-local candidate merge radius |

Pixel lengths need resolution-aware adjustment, and area scales quadratically.
Metre thresholds must reflect depth noise and geometry, not pixel scaling.
The small baseline also fixes original depth coverage at 80%, minimum interior
points at 20, and minimum internal-step pair count at 3. It is not an automatic
general-purpose calibration or segmentation framework.

## Implementation and evidence scope

Original code independently uses standard operations described by the official
[OpenCV contour tutorial](https://docs.opencv.org/4.13.0/dd/d49/tutorial_py_contour_features.html)
and [shape API](https://docs.opencv.org/4.13.0/d3/dc0/group__imgproc__shape.html).
No external detector implementation, data or learned weights are bundled.
Synthetic regressions check contracts, not field accuracy. Real-image performance,
registration error and overlap labeling are not established by this API. Detailed
comparisons and research records are maintained outside the repository.

## Bounded adaptive support (opt-in)

```python
detector = OverheadCornerDetector(support_policy="adaptive")
```

The default remains `"fixed"`. Both policies cap the long arc window at the
smaller of `edge_window_px` and 45% of either adjacent polygon edge. Fixed
policy uses a second window at 60% of that length. If only the short window
lacks `min_fit_support` points per side (default four), adaptive policy tries
80%, requiring the same configured minimum on each side and strictly fewer samples than the long window on each side. It never
extends the long window, crosses a polygon vertex, accepts a raw fallback,
relaxes line/depth thresholds, or makes an unsupported line fit certain.
Twenty-percent scale separation remains; it is an engineering constraint rather
than a proven confidence bound. Diagnostics record `adaptive_support_used`,
effective windows and support counts.


## Optional multiscale exterior proposals

The default `proposal_policy="single"` retains the original polygon proposal
path. Use `OverheadCornerDetector(proposal_policy="multiscale",
support_policy="adaptive")` to compare a small union of polygon approximations.
The four epsilon values are `polygon_epsilon_px * (0.5, 1, 2, 4)` in processing
pixels, not fractions of contour perimeter. Each comes from `cv2.approxPolyDP`
on the original external contour; no convex hull, corner labels, contour
completion or depth-hole filling is used.

Nearby proposals on the same contour are greedily merged within
`proposal_merge_px=3`. Representatives prefer convex near-right polygon geometry,
then usable adjacent-edge extent and angle closeness. This ordering uses only
mask geometry. Each retained result records the epsilon and scales represented.
After line refinement, coincident results are merged again within the same
contour, preferring accepted quality. This can suppress genuinely distinct corners
closer than the merge radius; scale pixel parameters with image resolution.

In `mode="rgb"`, the original convex/angle/edge-extent tests still select accepted
polygon candidates. In refined/depth modes, multiscale proposals may reach local
line fitting even when the initial polygon angle or short edge fails those tests.
The fitted edges must themselves be convex, near-right, sufficiently supported
and low-residual. Both window fits and the intersection-displacement check remain
required; angle-difference rejection is independently configurable. Depth quality checks remain unchanged.
An undefined angle at a degenerate line intersection is insufficient support.
Per-stage measurements remain available for candidates that are not accepted.

More proposals can recover exterior corners but also add false candidates,
particularly on small rounded boundaries at coarse epsilon. A polygon vertex or
accepted 2D angle is not proof of a physical right angle, single layer, or safe
grasp. Interior annotated corners absent from the supplied external silhouette
are outside this proposal mechanism. Upstream masks can themselves miss or alter
real boundaries. End-to-end segmentation and physical grasp accuracy are not
established by this mode. The independent grasp feature is unchanged.

For local evaluation, define any exterior-eligible label subset by distance to
the original supplied external contour, independently of predictions and epsilon.
Keep all-label scores and eligibility-tolerance sensitivity alongside subset
scores; do not use a fitted polygon to decide which labels count. Keep related
lighting/background views in the same scene group. Evaluation tools, source data
and before/after galleries belong outside this repository.

OpenCV documents epsilon as the maximum permitted curve-approximation distance:
[structural analysis / approxPolyDP](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html).
The bounded multiscale union and quality ordering here are independent engineering
choices, not an OpenCV semantic corner classifier or a learned model.

The optional policy is a recall/false-candidate tradeoff, not a new accuracy
claim. A small fixed development comparison found more accepted exterior
candidates but lower RGB-only precision; a rounded synthetic silhouette also
produced a false RGB-only corner that local refinement rejected. Keep the
single-scale default for compatibility. Inspect refined/depth evidence before
using multiscale candidates. Near-complete coverage by raw vertices does not
imply accepted-corner recall: angle, local support and depth checks can reject
those same vertices. No labels are used by the detector.


## Explicit recommended starting preset

The ordinary constructor remains single-scale/fixed-support with angle-stability
rejection. Existing callers of `angle_tolerance_deg` still control both polygon
and fitted angle tolerances unless either new override is provided. Existing
callers of `depth_step_m` still control boundary consistency unless
`boundary_mixing_m` is provided. New summary/diagnostic fields are additive.

For a deliberate tuning starting point, use:

```python
detector = OverheadCornerDetector.recommended_for_tuning(
    proposal_angle_tolerance_deg=25.,  # polygon ranking and RGB-only selection
    fitted_angle_tolerance_deg=25.,    # retain fitted 65--115 degree guard
    angle_stability_deg=10.,           # warning threshold, not rejection here
    max_intersection_shift_px=2.,
    min_fit_support=4,
    line_residual_px=1.,
    depth_residual_m=0.005,
    depth_step_m=0.015,
    boundary_mixing_m=0.015,
)
```

The preset changes only `proposal_policy="multiscale"`,
`support_policy="adaptive"`, and `angle_stability_mode="diagnostic"` relative to
the constructor defaults. Keyword overrides are allowed, including a return to
`angle_stability_mode="reject"`. It is a recommended **starting configuration**,
not an empirically optimal preset or a real-robot validation result.

Diagnostic mode still requires supported, nondegenerate long and short line
fits. It keeps local convexity, fitted angle, line residual, intersection shift,
fit-to-polygon displacement, visibility and depth checks. Missing short-window
support does not become an accepted single-window fallback. Minimum support now
controls both fit eligibility and bounded adaptive-window selection. Earlier
external support-threshold experiments which left adaptation fixed at four points
are not exact performance measurements of a different `min_fit_support` here.

`summary["tuning"]` records resolved parameters. Each candidate records
`stage_results`: polygon checks, local support counts and independently computable
line checks, then depth measurements/checks at the candidate's reported `xy`.
Line checks contain `value`, `maximum`, `passed`, and `mode`; a `None` pass value
means unmeasurable, not successful. Depth checks distinguish minimum coverage,
internal pair count, local residual, and boundary error. Uncomputable depth
measurements are absent, while available validity counts remain. These checks
remain available after earlier rejection; they are measurements, not a claim
that later-stage selection was reached. Local convexity is reported separately.
RGB-only mode records line diagnostics without using them for selection.

A failed diagnostic angle check adds
`warnings=["angle_stability_exceeds_diagnostic_threshold"]` even when another
check rejects the candidate. Aggregate `reasons` retain their existing semantics;
use independent check records for overlapping causes. An anomaly-free independent
depth probe now correctly reports `candidate_quality="accepted"`; older versions
reported `unknown` due to its initialization. This corrects diagnostic output,
not the legacy candidate acceptance path.

Suggested adjustment order:

1. Verify raw depth units, registration, original validity and the external mask;
   fix these before tuning thresholds. Keep scene/lighting variants together
   when separating tuning and confirmation data.
2. Scale pixel windows/epsilon/merge radius to image resolution and area
   quadratically. Keep generation fixed while comparing individual selection
   checks; changing proposal tolerance can change representatives/coordinates.
3. Start with the preset; compare angle stability as a warning versus rejection.
   Inspect fitted-angle error, residual, intersection shift and support together.
   Tightening support can change adaptive windows and fitted geometry, not just
   remove results. Retain convexity and nondegenerate-fit guards.
4. Tune internal depth steps, local residual and boundary mixing separately from
   sensor noise and registration evidence. Visible-corner labels alone cannot
   validate a fold/overlap gate. Do not relax missing-data checks to increase counts.
5. Measure localization, matched/unmatched candidates, coverage and unknown
   reasons on real robot data. Unmatched detections are not automatically false
   physical corners. Do not choose a preset just because it returns fewer unknowns.

## Thresholds, measurements and possible future ranking

The current measurements are 2D right-angle proximity, line-fit support and
residuals, local depth consistency, and image-space area/support. Specifically,
`contour_area_px` is the area enclosed by the exterior contour (which can include
holes); `interior_points` counts valid pixels in the eroded component-local disk.
Neither is a physical usable corner area or a gripper-clearance estimate.

**No table reference plane is supplied to this detector.** Even with intrinsics,
the plane is fitted to the local deformable-object patch. It does not measure
height above the table or proximity to a table plane. Reported outside-minus-inside
optical Z is not table-normal height. Use the independent `LiftDetector` with an
explicit table plane when that measurement is needed; it is not automatically
combined with Corner output.

There is no weighted quality score or calibrated probability. Proposal merging
uses lexicographic mask-geometry ordering; final merging prefers accepted status,
then lower line residual and coordinates. This is deterministic deduplication,
not a confidence model.

A future optional ranking could order candidates that have already met the
required validity/quality conditions, using angle error, supported local area
and consistency. Invalid input, degenerate fits or critical depth constraints
should not be compensated by a high angle score. Missing features should remain
explicitly unavailable; they should not silently become zero errors or be
renormalized into apparent certainty. Weights, normalization, required features
and missing-data behavior need a separate design and real-data validation.
This change exposes tuning and diagnostic evidence only; no new weighted
classifier is implemented, and no grasp/overlap correctness is guaranteed.
