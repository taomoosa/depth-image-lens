# Repeatable-pose gripper references

`ReferenceGripperSegmenter` is a controlled-scene alternative to supplying
perfect hand masks on every frame. It does **not** track a freely moving robot.
Its NumPy-only output is candidate segmentation, not contact or grasp truth.

## Capture and runtime contract

1. Fix the camera, depth mode, registration, crop and exposure. Capture an empty
   scene without the deformable object or robot. Capture the end effectors only
   at a repeatable pose, with no deformable object. Both are optical-axis Z.
2. Supply two coarse end-effector boxes, `left`/`right`, half-open `(x0,y0,x1,y1)`,
   and one boolean static background ROI outside both boxes. The boxes must not
   overlap and must avoid arm shafts. Inspect the generated reference masks.
3. `calibrate(empty_depth, gripper_depth, hand_rois, static_mask,
   pose_key="pose-A", registered_to_depth=True, ...)` creates `GripperReference`.
   Use a dictionary of these references for several **explicitly identified**
   poses. The caller selects the pose; the code never guesses it from a mask.
4. `segment(current_depth, reference, pose_key="pose-A",
   registered_to_depth=True, ...)` returns candidate `hand_masks`, `object_mask`,
   `uncertain_mask`, `observed_mask`, `foreground_mask` and quality summaries.
   If reference RGB was provided, runtime RGB is required. Declare `color_order`
   as `RGB` or `BGR`; uint8 HxWx3 must already be registered to depth.
5. Pass the candidate masks, observed mask and uncertainty to
   `IndependentGraspDetector`; keep completeness false until checked. For
   `ObjectOnlyGraspDetector`, pass gripper/uncertain masks separately so they do
   not inflate object width. Existing manually supplied masks remain a direct
   override through the detector APIs. Do not convert missing segmentation to
   a confirmed empty hand.

Depth arrays are HxW numeric ndarrays, scale is explicit metres/count for **all**
dtypes (`.001` default for uint16 millimetres, `1` for floating metres). Zero,
NaN and Inf are missing; invalid background does not establish foreground.
Reference depth validity is retained separately. A pixel missing in the gripper
reference cannot become object material just because runtime depth appears there.
It remains uncertain inside and outside search boxes. Per-side
`reference_search_valid_fraction` / `reference_search_unknown_pixels` use the
coarse search region, while `valid_fraction` describes observed **known hand**
pixels only; neither establishes a complete hand silhouette.
Array size equality alone does not establish registration. Capture again after
camera/table changes. Changing crop requires transforming **all** reference
arrays and ROIs consistently; separate calibration per processing grid is safer.
`hand_visibility=("visible", "partial")` is caller evidence, not inferred ID.
Left/right are supplied identities; image-left is not automatically robot-left.

## What is checked

Foreground is `empty - current > foreground_m`. A reference hand pixel must
match current depth and, when available, RGB. A static ROI monitors camera/scene
and illumination change. Each side has independent coverage/match checks. A
coarse-box check looks for unexpected gripper-like foreground outside the
reference silhouette. A failed side is withheld without erasing a valid opposite
side. Pose/static failures withhold the object candidate; outputs retain reasons.
Foreground present outside the hand boxes in the object-free reference is a
separate robot/clutter region. Matching pixels are returned as `robot_mask`, not
object material. Changed/occluded reference robot pixels remain uncertain. If
the robot match fraction is below `min_match_fraction`, object identity is
withheld while independent hand evidence remains available. This can also
abstain on a true object occluding the arm; it is not free-pose tracking.
Contact neighborhoods are uncertain, not silently promoted to contact proof.

| Parameter | Default | Meaning and adjustment order |
| --- | ---: | --- |
| `foreground_m` | .02 m | Minimum camera-Z separation; measure empty-scene noise first. |
| `match_depth_m` | .01 m | Reference/current depth tolerance; do not hide pose errors by widening it. |
| `match_color_distance` | 45 | Euclidean distance in raw 8-bit RGB, not perceptual Delta-E. |
| `static_depth_m` | .01 m | p90 static-ROI discrepancy ceiling. |
| `static_color_distance` | 35 | p90 static RGB discrepancy ceiling. |
| `min_pixels` | 20 px | Minimum reference/runtime support. Scale with processing resolution. |
| `min_valid_fraction` | .7 | Minimum observed fraction of each reference hand/static ROI. |
| `min_match_fraction` | .7 | Required match fraction among valid reference-hand pixels. |
| `max_unexpected_fraction` | .05 | Unexpected gripper-like pixels / reference-hand area. |

Defaults are provisional. Fix registration/pose/ROIs before thresholds. Inspect
validity, unexpected pixels and contact uncertainty before accepting masks.
This method cannot distinguish an occluding object that exactly reproduces hand
depth and color. Same-color surfaces at separated depth can be separated; similar
depth/color, patterned highlights, pose changes and occlusion can cause abstention
or semantic mistakes. Non-object moving clutter also becomes foreground.
Static checks do not prove alignment or detect every mismatch.

## Deployment acceptance

Evaluate with independently annotated hand/object masks, separate from reference
captures: hand recall, object false exclusion, object IoU, unknown fraction and
per-side downstream outputs. A provisional controlled-workflow target is hand
recall >= .98, object false exclusion <= .02 and object IoU >= .90 on **each**
confirmed aligned scene, with pose/lighting/occlusion negatives explicitly
reported. These are acceptance targets, not achieved real-world accuracy.
Current real examples lack independent empty/gripper reference pairs and human
segmentation/physical-grasp truth; practical readiness is **not established**.
Never transfer a start-image mask merely because camera pose is unchanged.

Run `python examples/example_reference_grasp.py`. It uses independent synthetic
empty, gripper-only and current arrays, not a current-image mask as a reference.
See [debug views](DEBUGGING.md) and [grasp contracts](GRASP.md).

## Other detectors and manual override

Use `segmentation.object_mask` as the optional selection for Lift, Corner or
DepthSurfaceDetector, keeping their own required calibration/registration.
Pass `segmentation.observed_mask & ~segmentation.uncertain_mask` as their
observation/validity mask where supported. These are provisional selections.
They do not establish a complete silhouette or hidden geometry. Do not pass a
reference-derived empty object mask as proof of absence when its status failed.

BridgeDrape can reuse coarse anchor boxes once their lower/contact edges are
checked: `detect_mask(current_depth, segmentation.object_mask, hand_rois,
depth_unit="mm", observed_mask=segmentation.observed_mask &
~segmentation.uncertain_mask, mask_complete=False)`. Broad calibration search
boxes are not automatically accurate contact bands; refine them once per pose,
or leave `hand_rois=None` to retain independent shape measurements with unknown
bridge. The existing depth-only auto-candidate route is also available. Neither
route requires a perfect per-frame hand mask. A manual verified selection can
be supplied directly to any detector instead of this reference workflow.
