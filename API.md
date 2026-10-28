# API and input contract

All generic scene targets are called **deformable objects**. The existing identifiers `work_roi`, `work_mask`, `work_pixels` and related reason strings are retained for API compatibility; they refer to the deformable-object search region, not a separate object category.

## Public API

```python
from depth_image_lens import BridgeDrapeDetector, BridgeDrapeResult

BridgeDrapeDetector(
    *, threshold_m=0.02, min_area=20, anchor_band_px=3,
    max_gap_px=0, sag_ratio=0.15, drape_ratio=0.25, static_hanging=True,
)

# Manual anchors:
detector.detect(
    depth, difference, hand_rois, *,
    depth_unit="m", difference_unit="m",
    difference_sign="background-current", difference_mode="signed",
    background_valid=None, observed_mask=None,
    arm_mask=None, work_roi=None,
    hand_visibility=("visible", "visible"), down=(0, 1), diagnostics=False, depth_jump_m=0.10,
    anchor_policy="complete",
)

# Shape-derived anchor candidates:
detector.detect_auto(
    depth, difference, *, upper_band=None, min_span_px=30,
    max_height_delta_px=12, depth_jump_m=0.10, down=(0, 1), diagnostics=False,
    **options,
)
```

The signatures above describe the API, not executable calls with placeholder variables. `detect_auto` accepts unit, difference-sign, validity, arm-mask and search-ROI options from `detect`; it rejects manual hand ROIs and hand-visibility declarations. See [AUTO.md](AUTO.md).

## Assumptions and inputs

| Input | Contract |
|---|---|
| `depth`, `difference` | Numeric HxW arrays with identical pixel coordinates, shape, unit and depth meaning. A fixed camera and unchanged background are required. |
| Units | Explicit matching `m` or `mm`. Threshold settings ending in `_m` always use metres. No automatic unit inference. |
| Difference | Default `difference_mode="signed"` retains physical sign handling. In `positive_foreground`, supplied positive values are candidates as-is; finite zero/negative values are observed non-candidates. No physical background is reconstructed. |
| Invalid data | Current depth <=0 or nonfinite, nonfinite difference, or a false optional validity mask is invalid in both modes. Only signed mode additionally checks the reconstructed background. In positive-foreground mode, finite zero declares observed exclusion/background; it does not repair missing measurements. |
| Masks | `background_valid`, `observed_mask`, `arm_mask`, `work_roi`: bool arrays of the same shape. Arm-mask true excludes a pixel. Search-ROI true includes a pixel. Missing/excluded pixels are not known empty space. |
| `hand_rois` | `{"left": (x0,y0,x1,y1), "right": (...)}`; x1/y1 exclusive. Lower edges should be near the candidate contact. A thin anchor band is retained while hand interiors are excluded. |
| `hand_visibility` | Two caller declarations: `visible`, `partial` or `hidden`. Valid depth alone does not establish unobstructed visibility. Mark known occlusion in `observed_mask` too. |
| `down` | Finite nonzero image vector, normalized internally. Default `(0,1)`. Anchor heights may differ in the image. Left/right are image-relative IDs, never automatically robot-hand IDs. |

This mode assumes a static deformable object suspended from above, with approximately equal hand heights for automatic candidates. It does not support arbitrary camera/background changes or infer physical contact from a closed gripper.

## Results

| Output | Meaning |
|---|---|
| `bridge` | `yes/no/unknown`: an observed candidate path reaches both anchors, no path exists in a complete visible search, or evidence is insufficient. |
| `large_sag` | `yes/no/unknown`: robust upper-envelope deviation from the sloping anchor chord, normalized by anchor span. It does not measure the lower edge or physical tension. |
| `drape` | `left/right/both/neither/unknown`: anchor-connected downward extent exceeds the configured span ratio. A one-sided result requires observable opposite anchor/search conditions. |
| `features` | Sag ratio, span coverage/length, per-anchor hanging extent/width, validity and gap information. Image lengths are pixels; ratios are dimensionless; unavailable values are `None`. |
| Masks | `foreground_mask`: valid positive differences. `work_mask`: candidate region after search/exclusions/area filtering. `invalid_mask`: missing depth/difference, not empty space. |
| `auto` | Present only for automatic candidates: stage, candidate ROIs/points, opposite search region and an explicit unverified-contact flag. |
| `validation_status` | `not_validated`: real accuracy has not been established; this is separate from an observation's `unknown` result. |

`result.to_dict()` produces JSON-safe values. `include_masks=True` includes bool lists and can be large. Individual outputs have independent conditions: for example, `bridge=no` can coexist with `large_sag=unknown`.

`drape=both` means enough extent belongs to components connected to each anchor. It does not imply two separate hanging pieces or symmetric local shape. A horizontal bridge with a long lower edge can have `drape=both` and `large_sag=no`.

## Tuning order

1. Verify registration, units, sign and invalid-data provenance before tuning thresholds.
2. Set manual hand boxes or the automatic upper band. Preserve the narrow anchor/contact band when excluding hands/arms.
3. Exclude upper arm paths with `arm_mask`; restrict the candidate region with `work_roi` when needed.
4. Set `threshold_m` from static-background noise. The default 20 mm is an example, not a universal validated value.
5. Set `min_area` and `anchor_band_px` for the image resolution. Each anchor requires at least three connected pixels.
6. Prefer `max_gap_px=0`. Optional 1-2 pixel jumps cross only observed pixels with locally continuous raw depth; they never fill invalid/excluded regions or large holes. Inspect `gap_used` and `inferred_gap_mask`.
7. Adjust `sag_ratio` and `drape_ratio` using representative labeled scenes. The conservative sag decision requires at least 95% observed span coverage; the independent measurement retains its numeric coverage and reference even when quality is limited. Apparent notches, openings, overlap and arm contamination can confound the upper envelope.

Connectivity is eight-neighbour. Visible paths around small holes can support a bridge; missing regions are never fabricated. Downward clipping can hide paths and yield unknown. Python component traversal can be slow on large images; crop consistently, updating every ROI/mask coordinate. No throughput guarantee is made.

## Real-scene validation still required

Record one fixed-camera empty background and static states showing a horizontal bridge, deep U, left-only drape, right-only drape, separate drapes and arms alone. Independently check masks, anchor candidates and actual contacts. There are currently no matched real background pairs in this project's tests, and no synthetic background is reported as a real evaluation.

Optional `diagnostics=True` on either method records the regions and envelope samples used. It does not change decisions. See [DIAGNOSTICS.md](DIAGNOSTICS.md) for array/JSON contracts and the dependency-free SVG example.

## Optional background subtraction

```python
from depth_image_lens import background_difference

difference, valid = background_difference(
    depth, background,
    depth_unit="mm", background_unit="mm",
    invalid_values=(65535,),  # Only if this is a sentinel for your sensor.
)
result = detector.detect_auto(
    depth, difference, depth_unit="mm", difference_unit="mm",
    difference_sign="background-current", background_valid=valid,
)
```

`background_difference(depth, background, *, depth_unit, background_unit, invalid_values=())` returns float64 difference and a bool valid mask. Both units must be the same explicit `m` or `mm`; there is no implicit conversion. Difference retains that unit. Casting before subtraction prevents unsigned wraparound, including negative differences. Nonfinite/nonpositive values, a declared sentinel in either image, or nonfinite subtraction make the difference NaN and mask false. Shapes must match and arrays must be real numeric HxW, not bool/object/complex. The helper performs no registration, camera-motion correction or hole filling.

The detector's main two-input API is unchanged. For precomputed differences in the general m/mm API, the CLI applies `invalid_values` only to current raw depth; difference zero is valid background; in signed mode negative differences must remain signed. Represent invalid precomputed differences with NaN or provide the relevant bool mask. Signed m/mm input requires `difference_unit` and `difference_sign`; background input instead requires `background_unit` and computes `background-current`. The explicit 16UC1 convenience described below uses clipped positives.

## Positive-foreground mode details

`detect(..., difference_mode="positive_foreground")` and `detect_auto(..., difference_mode="positive_foreground")` accept a masked difference without rebuilding background depth. The condition is `difference > 0` and `difference > threshold_m` after unit conversion. `threshold_m=0` is accepted; sag/drape ratios remain strictly positive. `features.difference_mode` records the mode for every result.

`difference_sign` belongs to signed mode. In positive-foreground mode, omit it (or pass `None`); the legacy default string `background-current` is tolerated for call compatibility but is not applied. `current-background` is rejected rather than silently negating a positive-foreground image. This mode makes no claim about which camera-depth subtraction created the positive values.

Known-zero pixels inside the search region can support a complete-search no-bridge result and the existing optional tiny observed-gap tolerance. They never become foreground pixels. Missing raw depth or false validity masks still prevent a fabricated empty-space or one-sided conclusion. An unseen opposite hand remains unknown even if the difference elsewhere is zero.

The CLI requires explicit matching `depth_unit` and `difference_unit` for precomputed inputs. It requires a sign for signed input, but not for positive-foreground input. With explicit m/mm units, raw `--background` input generates signed differences. The explicit `encoding="16UC1"` convenience below instead prepares clipped positives and retains their validity mask.

## 16UC1 ndarray convenience

```python
from depth_image_lens import positive_difference_16uc1

masked_difference, valid = positive_difference_16uc1(current_depth, empty_background)
result = detector.detect_16uc1(
    current_depth, masked_difference, hand_rois=None,
    depth_scale_m=0.001, background_valid=valid, upper_band=(8, 20),
)
```

`detect_16uc1(depth, difference, hand_rois=None, *, depth_scale_m=0.001, **options)` requires matching single-channel uint16 HxW arrays. It converts into separate metric float buffers and dispatches to positive-foreground auto mode (no hand_rois) or manual mode. It accepts noncontiguous views without modifying the supplied arrays. Both inputs use the same positive finite scale; unit/sign/mode overrides are rejected to avoid conflicting interpretations. No ROS installation, topic subscription or camera access is involved.

`positive_difference_16uc1(depth, background, *, invalid_values=())` casts to int32 before subtraction, clips negative values to zero, and returns uint16 counts plus a bool valid mask. Either zero or a declared sentinel in either source is invalid and yields zero with valid=false. The clipped image alone cannot recover why a zero occurred: retain and pass the validity mask. The signed float `background_difference()` helper remains available and unchanged in sign semantics.

`depth_pair_16uc1_to_meters(depth, difference, *, depth_scale_m=0.001)` exposes the same conversion if needed. Encoding is an explicit convenience opt-in; dtype never silently selects a distance scale for the general API. The CLI config equivalent is `encoding="16UC1"` plus optional `depth_scale_m`; it rejects mixed m/mm/sign configuration. All thresholds ending in `_m` remain metres.

## Table-relative lift

```python
from depth_image_lens import LiftDetector, LiftResult, CameraIntrinsics, TablePlane

LiftDetector(*, lift_height_m=0.03, near_table_m=0.01, height_error_m=0.005,
             min_pixels=20, min_component_pixels=12, min_lift_fraction=0.1,
             min_valid_fraction=0.7)
detector.detect(depth_m, intrinsics, table, object_mask, *,
                exclude_mask=None, observed_mask=None, whole_object_visible=False, diagnostics=False)
detector.detect_16uc1(depth, intrinsics, table, object_mask, *,
                      depth_scale_m=0.001, **options)
```

These are signatures, not executable placeholder calls. `detect` accepts optical-axis Z in metres; `detect_16uc1` accepts uint16 counts with the same explicit scale convention as BridgeDrapeDetector. Raw zero/nonfinite depth is missing, never observed empty space. The shared conversion reads noncontiguous views without modifying inputs. Radial range must first be converted to optical-axis Z; no distortion correction is performed.

| Input | Required meaning |
|---|---|
| `CameraIntrinsics(fx, fy, cx, cy)` | Rectified depth-grid pinhole calibration in pixels. `cropped(x0,y0)` adjusts an unscaled crop. Resize requires appropriately updated intrinsics; same-sized RGB is not proof of registration. |
| `TablePlane(normal, offset_m)` | Equation `normal.dot(point_m)+offset_m=0` in the depth-camera frame; both coefficients are normalized together. Orient the positive side above the table. A flipped sign, wrong plane, or camera motion invalidates the interpretation. |
| `object_mask` | Bool HxW mask of the visible deformable object in depth coordinates. Not restricted to the table's 2D projection. Remove robot/background contamination with `exclude_mask`. |
| `observed_mask` | Optional bool HxW evidence mask; false excludes interpolated, occluded or otherwise unobserved values from evidence. |
| `whole_object_visible` | Caller declaration including hidden/self-occluded parts, not simply an object bounding box inside the image. Defaults false. |

No physical height is inferred from a difference image: camera-Z difference is not table-normal height in an oblique view. A masked positive difference may help select candidates, but does not establish their semantic identity. Difference zero remains valid background/exclusion under the Bridge contract; Lift uses the supplied object/observation masks and raw depth instead. No background validity is needed if it did not contribute to the selected object evidence.

`LiftResult.to_dict()` gives strict-JSON-safe values: `state`, `reason`, `scope="visible_region"`, `whole_object_lifted`, `valid_fraction`, `elevated_fraction`, `near_table_fraction`, `height_quantiles_m` (p10/p50/p90), `largest_elevated_component_pixels`, and `validation_status="not_validated"`.

The visible state is lifted when a four-connected elevated component and elevated fraction exceed their thresholds. It is not_lifted when at least 80% of valid selected pixels are confidently near the plane; intermediate evidence is unknown. No hole filling occurs. The visible claim applies to the observed selected region, not to every selected pixel being elevated. Whole-object no requires a near-table component; whole-object yes requires sufficient finite observations and the minimum elevated-component size, and additionally requires the visibility declaration, complete valid object coverage, and every object pixel confidently elevated. Otherwise it is unknown. Near-table evidence is a proximity criterion, not a physical contact sensor.

Tune `lift_height_m` and `near_table_m` for the task, then `height_error_m` for combined depth/plane uncertainty, and pixel area/coverage thresholds for resolution and visibility. `height_error_m` is a caller-supplied assumption, not an estimated or guaranteed error bound. Uncertainty crossing the height threshold is retained. The default numeric thresholds are examples, not validated real-camera values. A wrong mask/plane can still produce a confident wrong result; no mask, plane, camera-motion or gravity validation is automatic.

This small implementation reuses the project's earlier independent LiftDetector and camera-frame geometry. The caller supplies a plane directly or uses the small SVD helper below; the earlier RANSAC framework and research data are not included. Synthetic regressions cover oblique height, table thickness, partial visibility/contact, missing depth, isolated pixels, count-scale equivalence and invalid geometry. They establish behavior only, not real-scene accuracy.

## Empty-table plane helper

```python
fit_table_plane(depth, intrinsics, table_mask, *, depth_scale_m=0.001,
                positive_direction=None, min_points=20, min_spread_m=0.01,
                residual_warning_m=0.005)
```

`depth` is registered, rectified optical-axis Z: numeric HxW ndarray, including uint16 16UC1. Scale is **metres per input count for every dtype**, never inferred from dtype: default 0.001 for millimetres, explicitly 1.0 for arrays already in metres. This shares scale validation/numeric conversion and deprojection with the existing package. Noncontiguous views are read safely; input arrays are not modified. No world/robot calibration is needed.

`table_mask` is a caller-selected bool mask on an empty-table image. Only finite positive selected depth contributes. Mark positive sensor sentinels, interpolated pixels, edges and clutter false yourself. Too few valid points, a line/too-narrow support, nonfinite geometry, or a non-unique normal raises `ValueError`; invalid samples among otherwise sufficient points are excluded. `min_spread_m` is the smaller in-plane RMS spread, not bounding-box width. This is a single least-squares SVD, with no RANSAC, outlier removal or automatic table recognition.

The default normal faces the camera origin, assuming the camera is on the above-table side. Alternatively set `positive_direction` to a nonzero vector toward the desired positive half-space **in the depth-camera frame**. A direction parallel to the fitted plane is rejected. Neither option identifies gravity or distinguishes a table from a wall; confirm the selected surface and sign. A changed camera pose/table requires a new fit.

`PlaneFitResult` returns `plane`, `point_count`, `valid_fraction`, absolute orthogonal `residual_median_m` / `residual_p95_m` / `residual_rmse_m`, two `in_plane_rms_spread_m` values, `orientation`, and `warnings`. `to_dict()` is strict-JSON-safe. P95 above `residual_warning_m` adds `large_plane_residual` but still returns the fit so callers can inspect or reject it. A small residual is not proof of correct ROI, accurate calibration, or a bound on unseen height error. Do not automatically use it as LiftDetector's `height_error_m`; that parameter must also account for sensor, pose, calibration and extrapolation error. Threshold defaults are engineering examples, not accuracy guarantees.

`python examples/example_lift.py` demonstrates uint16 empty image -> manual ROI fit -> current-image LiftDetector and prints both summaries. This uses a generated plane and is not an evaluation on a real empty-background pair.

## Bridge depth continuity and explicit gap inference

Both manual and automatic bridge modes accept `depth_jump_m=0.10`. It bounds raw camera-depth change on each traversed eight-neighbour edge, including diagonal edges and the intermediate pixels of explicitly enabled tiny gaps. The legacy conservative pass withholds a whole connected candidate containing an excessive jump. The independent feature pass instead removes that local transition and tests remaining observed routes, with no hole filling. Both decisions are retained. The threshold is an example and must match actual image resolution/surface slope/noise. This is not metric table height or semantic segmentation.

`max_gap_px=0` remains the default. With 1 or 2, a known-zero difference pixel remains observed non-foreground: a possible connection through it is explicitly **inferred**, never converted into observed foreground. Missing/arm-excluded/search-excluded pixels and larger gaps cannot be crossed. A zero-difference gap whose raw depth jumps to the background is rejected. `bridge_corridor_mask` contains only observed spanning-component pixels; `inferred_gap_mask` separately marks zero/low-difference pixels traversed by the accepted component search. It is not a shortest-path estimate. The bridge reason names this inference, and sag samples remain observed pixels only.

`depth_rejected_mask` identifies withheld manual candidates. Features include `depth_jump_threshold_m`, `depth_rejected_pixels`, and `inferred_gap_pixels`. Automatic `candidate_evaluations` records rank, span, area, gap use and selection/rejection decision. Its `depth_continuous` field retains the legacy whole-component check; `observed_route` and `rejected_local_edges` report the independent local-route check. A rejected candidate retains its proposed bands and feature measurements while other ranked candidates are considered. The search still considers at most four largest components and two widest eligible spans; rejecting the first for discontinuity now allows the second to be checked. Ranking is a bounded heuristic, not confidence or exhaustive object search. Same-shape, same-depth arms remain semantically ambiguous.

Continuity comparisons absorb only floating-point roundoff: per edge the numerical tolerance is `4 * float64_epsilon * max(abs(z1), abs(z2), depth_jump_m)`. Adjacent and optional gap edges use the same comparison. This is not a sensor uncertainty allowance; the configured depth threshold otherwise remains unchanged.


## Registered color assistance

`SeededColorAssistant(*, max_color_distance=60., min_color_margin=15., boundary_guard_px=1)`

`segment_16uc1(depth, difference, color, *, object_seeds, hand_seeds, color_order, registered_to_depth, depth_scale_m=.001, threshold_m=.02, observed_mask=None, background_valid=None, color_valid_mask=None)`

`segment(color, foreground_mask, *, object_seeds, hand_seeds, color_order, registered_to_depth, valid_mask=None, color_valid_mask=None)`

Returns `ColorAssistResult(object_mask, hand_mask, uncertain_mask, summary, hand_exclusion_mask=None, appearance_uncertain_mask=None)`.
The last two masks are populated by the assistant and separate exclusion/class
ambiguity from the eroded candidate interiors. See the experimental integration
example for preserving original depth validity and reporting semantic abstention.
Only the depth-count convenience path requires a scale; no world/robot calibration
is needed for color segmentation. Color/depth pixel registration is mandatory.
See [RGBD.md](RGBD.md) for masking, seed, ambiguity and integration contracts.

## Optional overhead corners

`OverheadCornerDetector.detect(color, depth, *, color_order, registered_to_depth,
object_mask=None, background_color=None, depth_scale_m=.001, valid_mask=None,
intrinsics=None, mode="depth", diagnostics=False)` returns `CornerResult`.
See [CORNER.md](CORNER.md) for the optional dependency, full input contract and
candidate/unknown semantics. This calibration contract is independent of BridgeDrape.

## Optional local observed anchor support

Manual `detect` / `detect_16uc1` accepts `anchor_policy="local_observed"`.
The default remains `"complete"` for the conservative pass. The local option
requires `max_gap_px=0` and an observed 3x3 foreground patch at both anchors.
The independent pass now also retains positive geometry from such local support
when complete-hand checks fail. Partial visibility warns rather than suppressing
all features; ROI and caller exclusions remain applied. Negative claims still
require complete evidence. No sensor depth or physical contact is inferred.
See [RGB-D input and uncertainty contracts](RGBD.md) for limitations.


## Feature-wise continuation

`BridgeDrapeResult.stage_results` separates observed bridge, per-band drape,
silhouette morphology, hanging extent, sag, inferred support hypotheses and grasp
certainty. Each entry carries `computed`, `inferred`, or `insufficient_input`,
measurements and applicable warnings. The static hanging assumption defaults to
true and can be disabled with `BridgeDrapeDetector(static_hanging=False)`.
See [RGBD.md](RGBD.md#independent-features-and-evidence-quality) for changed root
semantics and inferred-reference limits. `features.conservative_decisions` retains
the earlier blanket-gated decisions. Physical grasp remains unknown.

Lift keeps finite height statistics under low coverage; Corner candidates retain
local line/depth measurements even when polygon acceptance fails. Their decision
quality checks remain separate from numerical computability. No input validation
errors are swallowed, and no missing geometric samples are synthesized.

## Result debugging and additional inspection APIs

All public spatial result types offer explicit `render_debug` / `save_debug`;
see [DEBUGGING.md](DEBUGGING.md) for RGB/BGR, depth-only and overwrite contracts.
`fit_table_plane(..., diagnostics=True)` also records its selected/valid ROI.
No renderer starts acquisition or saves implicitly.

- [ReferenceGripperSegmenter](REFERENCE.md): repeatable-pose empty/gripper-only
  calibration with coarse setup boxes and runtime quality checks.
- [DepthSurfaceDetector](SURFACE.md): height/reference separation and independent
  local shape; physical layers remain unknown.
- [ObjectOnly partial gripper context](OBJECT_GRASP.md): optional exclusion and
  uncertainty masks plus visible pixel extent ratios.

## Overhead internal boundaries

`OverheadBoundaryDetector` independently measures color foreground, local darkness,
depth steps and relative-depth islands. See [BOUNDARY.md](BOUNDARY.md) for inputs,
parameter units, stage uncertainty and public `render_debug` / `save_debug` methods
on `BoundaryResult`. `examples/example_boundary.py` is a minimal in-memory example.
Color evidence and relative relief do not establish wrinkles, layers or graspability.
