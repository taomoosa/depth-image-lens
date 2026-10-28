# Overhead color and relative-depth boundaries

`OverheadBoundaryDetector` provides **image evidence**, not semantic wrinkle,
layer-count or grasp classification. It is an independent optional OpenCV feature;
install the `boundary` extra (or the existing `corner`/`debug` extra). Existing
BridgeDrape, grasp, Lift and Corner behavior is unchanged.

```python
from depth_image_lens import OverheadBoundaryDetector

# color: uint8 HxWx3 RGB, depth: registered HxW optical-axis Z uint16.
# Default scale is 0.001 metre/count; dtype alone does not determine units.
detector = OverheadBoundaryDetector(color_distance=30, depth_step_m=0.01)
result = detector.detect(color, depth, color_order="RGB",
                         registered_to_depth=True, depth_scale_m=0.001)
summary = result.to_dict()  # json.dumps(summary, allow_nan=False)
image = result.render_debug(color, color_order="RGB")
result.save_debug("debug.png", color, color_order="RGB")  # explicit write
```

`render_debug` returns uint8 in the requested RGB/BGR order, without changing inputs.
`save_debug` writes PNG, requires an existing parent directory and refuses existing
files unless `overwrite=True`; detection never writes. Masks and numeric maps remain
available even with `diagnostics=False`. The image includes separate masks, stage
reasons, parameters, darkness/gradient and signed residual heatmaps. Magenta numeric
pixels are unavailable, not zero measurements. Edge-band contours are not skeletons
or subpixel centerlines.

## Select the object domain

Choose one segmentation source below. An explicit `object_mask` may also be passed
with `background_color`: the mask wins and the unused reference is reported as ignored.

- `background_color`: independently acquired empty-scene image, same uint8 format,
  camera pose, pixel grid and color order. Per-pixel encoded-Lab distance selects
  changes. Optional declared translation, illumination correction and quality checks
  are described below; shadows and moving hands can also be selected.
- `table_mask`: bool HxW samples of **currently visible table**. A palette of up to
  16 quantized table colors is estimated. No fixed hand or object color is assumed.
- `object_roi=(x0,y0,x1,y1)`: coarse integer rectangle in processing pixels.
  GrabCut fits foreground/background color mixtures; outside the rectangle is assumed
  background. All probable labels inside remain marked uncertain. The rectangle must
  include the whole visible target and leave real background outside. It is not a
  perfect mask requirement, but clutter/same-color/cropped targets can still fail.
  OpenCV initialization can vary; the external replay tool sets a process RNG seed
  and records it. The library does not reset global OpenCV RNG state.
- `object_mask`: bool HxW visible deformable-object region supplied by the caller.
  This overrides segmentation only, not the boundary/depth calculations.
- None: outer `border_px` pixels are assumed to sample table. This assumption is
  reported and is unsafe when the object or other items occupy the image boundary.

The Lab representation is OpenCV's 8-bit encoding: `color_distance` is Euclidean
encoded-Lab distance, **not standardized Delta E**. Near-threshold pixels are
reported in `uncertain_mask`; retained foreground may overlap this mask. Same-color
objects can disappear completely. No separation means insufficient evidence, not
proof of an empty scene. Table texture, shadows and hands are not semantically
removed. Multiple components remain separate for depth fitting. Cropped selections
are flagged; their exterior is only the observed exterior. Neither automatic mode
is general semantic segmentation. Inspect the mask before interpreting boundaries.

For approximately uniform, chromatically distinguishable table/object pairs, see
[optional shadow hypotheses](BOUNDARY_SHADOW.md). This opt-in mode also needs a small
object-color sample; it does not guarantee separation of brightness-only colors.

For independent brightness/chromatic sensitivity without an object seed, see
[background tolerances](BACKGROUND_TOLERANCES.md). Equal tolerances reproduce the
legacy foreground decision; larger brightness tolerance also risks missing dark objects.

## Optional empty-background workflow

Capture a separate empty scene with the same camera, crop, pixel grid and table;
remove the deformable object, hands and other foreground items. Lock exposure and
white balance when practical. Reference images from a different object arrangement
are **not** empty-scene references. The detector does not establish emptiness or
recover an object that has the same appearance as its background.

```python
import numpy as np
from depth_image_lens import OverheadBoundaryDetector

# Both uint8 HxWx3 arrays use RGB. Current depth remains optional.
# Mark a stable, currently visible table patch in current-image coordinates.
check = np.zeros(color.shape[:2], dtype=bool)
check[10:30, 10:40] = True  # illustrative location; inspect it in your own image
model = OverheadBoundaryDetector(background_compensate=True,
                                 color_distance=30, background_check_p95=20)
result = model.detect(color, depth, color_order="RGB", registered_to_depth=True,
                      background_color=empty_color, background_check_mask=check,
                      diagnostics=True)
print(result.stages["background_reference"])
```

The check patch must contain at least 25 valid pixels of unchanged table in both
images. It must not contain the object, a hand, a moving shadow or an occluder.
`background_compensate=True` estimates a single median current-minus-reference Lab
channel offset on this patch. It is an approximation for modest global illumination
change, not spatial shadow removal or exposure calibration. The offset norm must
not exceed `background_max_offset=40`; the corrected patch distance p95 must not
exceed `background_check_p95=20`. Without compensation, the same p95 check applies
to raw distance. These units are encoded-Lab units, not metres or calibrated Delta E.
A passing patch does **not** validate other regions or certify an empty reference.
The reported `compensation_applied` describes the distance calculation; if the
reference is rejected, that corrected distance is diagnostic only.

Optional per-call controls:

- `background_valid_mask`: strict bool HxW in **reference coordinates before shift**.
  False pixels are unavailable, remain uncertain and cannot create reference-based
  foreground. Supply it for known occluded, missing or unusable reference regions;
  do not mark them as observed empty table.
- `background_shift_xy=(dx,dy)`: known integer translation of the reference into
  the current grid, in processing pixels. Positive dx moves reference content right,
  positive dy moves it down. The validity mask moves with it; uncovered borders are
  unknown. No translation is estimated automatically. Rotation, scale, perspective
  or camera-pose change requires caller registration or a new capture. Do not tune
  the shift solely to make the target object disappear.
- `background_check_mask`: strict bool HxW in **current coordinates after shift**.
  Optional when compensation is off; without it the reference is unchecked.
  A global appearance change can then select the entire frame as foreground.
  Use a representative stable table check in operation; unchecked candidates are
  not validated segmentation.
- `background_color=None`: retains the original no-reference behavior; reference-only
  options are unused. An explicit `object_mask`, including an empty one, overrides
  the reference and is never expanded because reference data is unusable.

`stages["background_reference"]` reports `unchecked_candidate`, `checked_candidate`,
`rejected`, `insufficient` or `ignored`, with offsets, validity, p95 distances,
thresholds and reasons where computable. Malformed optional references, inadequate
check support or failed quality limits use the existing border-table color fallback,
mark its entire mask uncertain and continue color/depth processing. This fallback
is not a verified object mask: inspect it or supply an explicit mask/independent
`depth_roi`. A wholly unavailable but correctly shaped reference produces no
reference foreground. Automatic empty segmentation may still return explicitly
unscoped depth evidence, as documented below. Same-image references can pass a clean
check while concealing all objects: input provenance remains essential.

With `diagnostics=True`, `result.diagnostics` includes `background_valid_mask`,
`background_check_mask` and `background_distance` (NaN where unavailable), when they
can be computed. The debug image adds a background-distance heatmap and reference
masks. `to_dict()` remains strict-JSON-compatible; numeric arrays are exported
separately. With diagnostics off, stage summaries remain but reference maps are not
retained. Detection does not mutate either input or write files.

`examples/example_boundary_background.py` demonstrates these controls using a small
synthetic pair. Existing real datasets used during development have no verified
independent empty-scene pairs. Reconstructed backgrounds and nonempty real reference
pairs test behavior and failure modes; they do not establish real paired-reference
segmentation accuracy. Acquire and label a small independent pair set in the target
camera setup before relying on this mode.

## Independent stages and outputs

- `object_mask`, `exterior`: selected foreground and external contour vertices in
  processing-image `(x,y)` pixels. Holes are preserved in the mask, but `exterior`
  contains external contours only.
- `darkness`: grayscale closing minus original grayscale, intensity units 0..255.
  Its local background is the closing envelope over `color_window_px`. A broad dark
  area wider than this window can have a dark interior but zero black-hat response.
- `gradient`: magnitude of 3x3 Sobel derivatives divided by 8, intensity per pixel.
  It also detects bright transitions. `color_boundary_mask` selects darkness OR
  gradient above threshold and removes small connected bands. `lines` contains
  each band's contours, area, mean/p95 darkness and p95 gradient. Prints, seams and
  shadows can produce exactly this evidence; they are not confirmed folds.
- `depth_boundary_mask`: adjacent observed depth samples whose detrended camera-Z
  residual difference exceeds `max(depth_step_m, 2*depth_noise_m)`. Missing pixels,
  object exterior and the excluded boundary band do not create steps. No holes are
  filled. Pixel adjacency is independent of physical pixel size.
- `depth_residual_m`, `islands`: fit a robust component plane using affine inverse
  depth, valid for a pinhole view of a plane. Positive residual is toward the camera.
  Without `intrinsics` the residual is camera-Z metres; with `CameraIntrinsics` it
  is signed normal distance in metres. **This is not table-relative height.**
  `depth_step_m` always uses detrended camera-Z differences. A trim-based fit assumes
  a dominant approximately planar baseline and can fail on broad curvature or a
  minority baseline. Each component reports its fit/support and failure separately.
- `island_mask`: connected residual patches exceeding the noise/height threshold,
  within area/fraction limits and fully surrounded by observed pixels of the same
  component. `islands` also retains rejected patches with sign, area, median residual,
  containment, component fraction and boundary-step support. An observed surrounding
  annulus excludes the candidate from a second inverse-plane fit; local contrast and
  support are returned and insufficient/weak context retains the patch as unknown. Boundary support is a
  measurement, not an acceptance requirement. A smooth mound or one suspended sheet
  can qualify; isolated noise is limited by area but correlated noise remains possible.
- `agreement_mask`: color bands within `association_px` Chebyshev pixels of a depth
  step. `color_agreement_fraction` is an overlap fraction, **not a probability**.
  Color-only evidence is retained when depth is unavailable. Coincidence can still
  be a shadow/print on unrelated relief. `layer_status` is always `unknown`.

Depth is optional. Valid values are finite positive optical-axis Z; 0/NaN/Inf and
`observed_mask=False` are unavailable. Masks are strict bool and all arrays must share
shape/crop/registration. An unusable depth shape/type/scale, observation mask, intrinsics or registration
contract is reported as `depth.status=insufficient`, `reason=invalid_depth_contract`
with an error string; color processing continues. Invalid required color, explicit masks, table samples, rectangle selections or
constructor settings raise clear errors. Invalid optional background images or their
reference masks/transforms are isolated as described below. Local fit failure does not stop
color or other connected components.

`depth_roi` can independently restrict depth analysis when segmentation is uncertain.
If automatic color selection finds no object and there is no depth ROI, depth
analysis covers the full frame with the explicit domain `full_frame_without_object_identity`. The depth stage sets `unscoped_evidence=True` and
`object_identity_available=False`; those results may include object/table exteriors
and must not be described as internal object boundaries. An explicit empty
`object_mask` or `depth_roi` remains empty. Area-filtered supplied masks never expand
to the frame; filtered pixels are marked uncertain. `stages` separates candidate/computed/
insufficient outputs; a computed stage can correctly have zero candidates. No camera
intrinsics are required for camera-Z evidence. If supplied after crop/resize, update
principal point and focal lengths. Radial range is not optical-axis Z.

## Tuning order and reproducibility

All defaults are provisional, not sensor-calibrated accuracy guarantees. Save
`detector.parameters` as JSON, reconstruct with `OverheadBoundaryDetector.from_config`.

| Parameters (defaults) | Unit / role |
|---|---|
| `color_distance=30`, `color_margin=8` | encoded Lab selection and ambiguity band |
| `border_px=5`, `min_object_area=100` | table border width / foreground component px area |
| `boundary_guard_px=3` | excluded object/exterior pixel band |
| `color_window_px=9` | odd 3..101 closing window, pixels |
| `darkness_threshold=12`, `gradient_threshold=12` | intensity / intensity per pixel |
| `min_line_pixels=6` | minimum connected evidence-band area |
| `depth_step_m=0.01`, `depth_noise_m=0.002` | adjacent step / declared noise floor, metres |
| `island_height_m=0.01`, `min_island_area=25` | relative residual / connected px area |
| `max_island_fraction=0.35` | maximum patch fraction of observed component |
| `island_context_px=5` | surrounding annulus radius for local contrast verification |
| `grabcut_iterations=3` | 1..10 iterations for optional coarse-rectangle segmentation |
| `association_px=3` | color/depth spatial association tolerance |
| `background_compensate=False` | opt-in declared-table median Lab offset correction |
| `background_check_p95=20`, `background_max_offset=40` | check residual and offset norm, encoded Lab units |

1. Verify optical-Z units, registration, table/reference assumptions and object mask.
2. Tune color selection before interpreting internal lines; use an independent table
   ROI or empty reference if border sampling includes the object.
3. Set window/band sizes for the processing resolution, then darkness/gradient thresholds.
   A resized image requires pixel lengths and areas to change (area scales quadratically).
4. Measure sensor noise on a known plane; set depth thresholds above noise. Metric
   thresholds do not scale with image resolution. Check the plane fit and coverage.
5. Tune patch size/fraction and association tolerance, inspect missingness and negatives.
   Do not improve apparent recall merely by swallowing missing depth or large domains.

`examples/example_boundary.py` generates a small analytic scene and optionally saves
its debug image. Dataset loaders and the local human comparison tool intentionally
live outside Git; their evaluation report records source licenses, hashes, split,
resolution, configs and exact commands. No pretrained model, external source code,
images or dataset-specific paths are bundled here.

## Method scope

OpenCV rectangle-based GrabCut is documented at
https://docs.opencv.org/4.x/d8/d83/tutorial_py_grabcut.html . It adds spatial constraints
for textured tables; it does not establish object identity.
OpenCV morphology and filtering definitions are documented at
https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html and
https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html . This independent
implementation uses black-hat/gradient measurements instead of treating Canny edges
as wrinkles. Inverse-Z detrending follows the pinhole plane equation; it does not
recover hidden surfaces or distinguish aligned layers. Real supplied-mask consistency
and qualitative depth evidence are different from annotated wrinkle/overlap accuracy.

## Optional continuity and topology

Enable `trace_boundaries=True` for independently adjustable weak continuation,
short inferred gap connections, ordered centerlines, corners and junctions.
See [BOUNDARY_TOPOLOGY.md](BOUNDARY_TOPOLOGY.md) for parameter effects, coordinate
conventions, evidence provenance and limits. Original measured masks and metrics
remain unchanged; the graph is not physical wrinkle identity.
