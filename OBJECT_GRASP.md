# Object-only inferred holding

`ObjectOnlyGraspDetector` implements an explicit user-selected rule: a sufficiently
elevated static deformable object is inferred held; a sufficiently wide elevated
connected region is inferred bilaterally supported. Hands need not be visible.
This class is separate from `IndependentGraspDetector`, whose local hand-contact
semantics and defaults remain unchanged.

```python
from depth_image_lens import ObjectOnlyGraspDetector

result = ObjectOnlyGraspDetector(min_lift_m=.03, min_width_px=40).detect_16uc1(
    current_depth, object_mask, difference=masked_difference,
    depth_scale_m=.001, background_valid=original_background_valid,
    diagnostics=True,
)
print(result.to_dict())
```

The two depth arrays are matching HxW uint16 NumPy arrays. The bool `object_mask`
must select the deformable object and exclude arms, occluders and background.
It may be manually supplied or come from caller segmentation/color assistance;
this class does not discover semantic object identity. Foreground difference alone
includes arms and is not a substitute for that selection. No hand masks, RGB,
robot calibration or gripper identity are required.

## Reference routes

- `detect_16uc1(depth, object_mask, *, difference=None, depth_scale_m=.001,
  **options)` converts explicit metres/count without changing inputs. Noncontiguous
  views work. General `detect(depth, object_mask, *, depth_unit, ...)` requires
  `m` or `mm`; any difference uses the same unit.
- Difference route: supply signed or clipped-positive **empty-background minus
  current** optical-axis Z. Positive separation is a task-specific proxy for
  lifting, **not table-normal height**. A resting object above a floor/background,
  motion toward the camera, or a wrong background can satisfy this proxy. Use a
  fixed registered empty background and verify that this proxy describes the task.
  `min_lift_m` here thresholds camera-Z separation, not vertical height.
- Plane route: omit difference and provide `intrinsics=CameraIntrinsics(...)` and
  `table=TablePlane(...)`. This measures signed table-normal height in the depth
  camera frame. Use `fit_table_plane` on an empty-table ROI if needed. Rectified
  optical-axis Z, correct intrinsics and above-table positive sign are required;
  radial range is unsupported. No external robot/world pose is needed.
- Neither reference: return `unknown` with `missing_reference`; absolute camera
  depth or image bounding-box height alone never implies lifting. Mixing routes
  or supplying a plane without intrinsics is an input error.

Raw zero/nonfinite depth, `observed_mask=False`, and nonfinite reference values
are unobserved. Difference mode also respects `background_valid=False`. Retain
that mask when clipping invalid background pixels to zero. Finite difference zero
is valid exclusion/non-elevation, never a missing sample or evidence of physical
absence; no physical background is reconstructed from a masked difference.
Positive sensor sentinels must be withheld through validity masks. Masks and
references must already share the same registered/cropped pixel coordinates.

## Rules and outputs

`held` is `held`, `not_held` or `unknown`; `bilateral` is `bilateral` or `unknown`.
They are explicitly rule-based **inferences**, not observations or probabilities.
`physical_grasp` is always `unknown`; scope is the visible selected region.
A single hand stretching an object, third-party supports, folds, or segmentation
errors can mimic the bilateral rule. Failing the width rule does not mean one hand.
The API does not identify left/right physical hands or require a bridge.

Default provisional parameters (tune on actual sensor/task data):

| Parameter | Default | Meaning |
|---|---:|---|
| `min_lift_m` | .03 | Minimum reference separation after error allowance |
| `reference_error_m` | .005 | Caller-assumed reference error, not estimated guarantee |
| `min_pixels` | 20 | Minimum pixels in one elevated connected region |
| `min_lift_fraction` | .1 | Minimum size of that region / valid selected pixels |
| `min_valid_fraction` | .7 | Minimum reference coverage within the selected mask |
| `depth_jump_m` | .1 | Maximum adjacent raw Z jump within a region |
| `min_width_px` | 40 | Minimum contiguous horizontal run width |
| `min_width_rows` | 3 | Consecutive rows supporting that width in the same region |

Connectivity uses four neighbors and actual observed elevated pixels. No holes,
exclusions or missing values are filled. An excessive adjacent depth jump cuts
that transition. Width is the widest horizontal run that is supported across the
configured consecutive rows, not a bounding box or the sum of separate islands.
Multiple qualifying components are reported separately; a bilateral claim must
be supported by one component alone. The chosen component ID is recorded.
Width is in processing **pixels along image x**, not physical length or a detected
support span. Align the image's lateral direction appropriately and scale pixel
parameters with resolution (areas quadratically). Large camera roll, perspective,
folding and mask errors can distort it. No metric-width mode is implemented.

Cropping does not invalidate sufficient visible positive support. Its width is a
lower bound only; a narrow crop cannot prove unilateral support or absence.
`whole_object_visible` and `mask_complete` default false. `not_held` is available
only with a plane reference, both declarations true, no mask contact with the
image boundary, enough selected pixels, complete finite observed reference, and
all selected points within `min_lift_m` of the table including the error allowance.
It means **not sufficiently elevated under this rule**, not mechanically ungrasped:
an object held while resting on a table can satisfy it. Difference-only zeros or
an empty object mask never produce a negative claim.

`metrics` includes reference type, counts/coverage, separation quantiles, component
areas, width measurements and chosen ID. `thresholds`, `reason`, `bilateral_reason`
and `warnings` explain decisions. Metrics remain available if insufficient
coverage prevents a claim. With `diagnostics=True`, `valid_reference_mask`,
`elevated_mask` and `component_labels` are returned separately. `to_dict()` omits
arrays and supports strict JSON. `validation_status` remains `not_validated`.

Run `python examples/example_object_grasp.py`. Regression scenes are synthetic,
not real-data grasp accuracy. No data, model, API or robot operation is needed.
