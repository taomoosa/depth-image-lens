# Outer-edge bulge evidence

`OuterEdgeBulgeDetector` finds a camera-near outer object band relative to the
nearby **inward surface of that same object**. It does not require the candidate
to be surrounded by object pixels. This is separate from
`OverheadBoundaryDetector.islands`: existing island semantics are unchanged.

```python
from depth_image_lens import OuterEdgeBulgeDetector

model = OuterEdgeBulgeDetector(min_height_m=.005, band_px=16, context_px=24)
result = model.detect(
    depth, object_mask,
    depth_scale_m=.001,       # mm input; use 1.0 for metres, including float arrays
    registered_to_mask=True,
    observed_mask=observed_mask,   # optional: where the mask/scene is observed
    depth_valid_mask=depth_valid_mask,  # optional: sensor-specific invalid samples
    diagnostics=True,
)
print(result.state, result.metrics)
for candidate in result.candidates:
    print(candidate['point_xy'], candidate['median_height_m'], candidate['region_ids'])
result.save_debug('edge-bulge.png', color, color_order='RGB')
```

Inputs are HxW optical-axis Z and a same-size boolean object mask on the same
pixel grid and acquisition. There is no automatic registration. Fixed overhead
view is assumed; colours are only optional debug backgrounds. Missing/nonfinite
and nonpositive Z are invalid. Mask positive sensor sentinels with
`depth_valid_mask=False`. `observed_mask` describes observation of the silhouette,
not just valid object depth; setting all outside-object pixels false makes the
silhouette unobserved. No outside-object depth is needed. Select one component
with your existing outline/component mask if desired; otherwise components are
analysed independently. The input mask is never eroded or modified in the result.

## Algorithm and interpretation

1. Find each component's external silhouette. Hole boundaries are not external
   edges. Define an inward band by Euclidean distance from silhouette pixel
   centres (zero on the contour). Reject guard pixels near any object boundary,
   and bands within `band_px` of a silhouette pixel on the image frame or next to
   unobserved pixels. This does not recover a physical edge outside the image.
2. Split the evaluation band into grid cells and connected pieces. For each,
   expand through observed, valid-depth pixels of that component (8-connected steps) up to
   `context_px`. Reference pixels must be farther inward than
   `band_px + context_gap_px` and no farther than that plus `context_px`.
   Context never fills holes or jumps across disconnected components or missing-depth strips. Around a
   concavity it can still reach another arm along a sufficiently short observed
   connection; local fitting cannot establish material identity.
3. Require local depth agreement, sufficient context coverage and points. Fit
   `1/Z = ax + by + c` robustly to valid inward samples. Reject insufficient
   robust support and references with excessive p90 absolute residual. A broad
   curved reference can therefore yield unknown instead of a candidate.
4. Without intrinsics, relative height is predicted camera Z minus observed Z.
   With `CameraIntrinsics`, it is signed distance to the fitted plane in metres.
   Positive means toward the camera. Accept supported band pixels at least
   `max(min_height_m, 2*depth_noise_m) + reference_error_p90_m` above that reference.
5. Group connected positive pixels per component, requiring minimum area and
   at least one pixel within `attachment_px` of the outer contour. This allows
   ignoring a narrow mixed-depth rim without requiring unreliable rim samples.
   It is evidence consistent with edge attachment, not proof that an unobserved
   gap physically connects to the edge. No hole filling, expansion or joining
   across missing candidate samples is performed.

The reference is from the **current inward surface**, not an empty image/table,
absolute world height or layer count. An elevated but flat object is negative.
A dip is not a positive bulge. Broad bulges extending into the reference can be
underestimated or absorbed entirely. Piecewise grid fits can introduce seams;
results depend on image resolution and grid placement. A coherent depth bias or
wrong supplied silhouette can mimic a bulge. There is no calibrated probability,
material/fold classification or physical grasp guarantee. Intrinsics cannot
resolve missing support or sensor bias. Distinct fits need not form one global
surface. A candidate touching the band limit has unmeasured full inward extent.

## Parameters

All defaults below are constructor defaults. Pixel values refer to the processing
image, not resized debug panels. Integer parameters reject bools/nonintegers.
All real values must be finite; bools are rejected.

| Name | Type, default, allowed range | Meaning and adjustment |
|---|---|---|
| `band_px` | int 16, 2..128 px | Maximum inward contour distance evaluated. Increase for broader bands; moves the reference inward, can leave narrow objects without support. |
| `edge_guard_px` | int 2, 0..32 px | Ignore the closest mixed-depth rim, including holes. Evaluation requires outer distance > guard and distance to any excluded object pixel centre > guard+1. Larger reduces noise but misses narrow bulges. |
| `attachment_px` | int 6, 1..128 px | Accepted positive region must approach this close to outer silhouette. Require `edge_guard_px < attachment_px <= band_px`. Larger admits more detached-looking ridges. |
| `cell_px` | int 24, 8..128 px | Grid size for local fits. Larger pools more context but may mix different slopes/arms; smaller can lack support and increase computation/seams. |
| `context_px` | int 24, 2..64 px | Geodesic expansion steps and inward reference-band width. Larger finds more support but can include other relief. |
| `context_gap_px` | int 3, 0..32 px | Gap after evaluation band before inward reference begins. Larger avoids bulge tails in baseline but increases extrapolation and missing context. |
| `min_context_points` | int 30, 12..100000 pixels | Minimum supported context pixels before and after robust trimming. Larger is more conservative. |
| `min_context_fraction` | real .7, (0,1] | Supported, valid-reachable pixels divided by context reachable through the observed mask (before depth-validity filtering). Larger rejects more missing/noisy reference. It does not certify coverage of unobserved space. |
| `max_reference_error_m` | real .003, >0 m | Maximum p90 absolute context-to-plane residual. Smaller rejects curved/noisy references; larger accepts less reliable baselines. Actual error also increases the acceptance threshold. |
| `min_height_m` | real .005, >0 m | Minimum positive relief; effective threshold has noise and measured-reference-error floors. Smaller detects lower bands but increases false candidates. |
| `depth_noise_m` | real .002, >=0 m | Noise allowance, twice this is a threshold floor. Set from sensor measurements; lowering height alone cannot cross this floor. |
| `local_radius_px` | int 1, 1..5 px | Square neighbour radius for depth agreement in this component. Larger rejects short spikes but can reject genuine steep/narrow relief. |
| `min_support_fraction` | real .6, (.5,1] | Required agreeing neighbours / component pixels in neighbourhood; also at least 5 agreeing samples including the centre. Missing depth fails agreement. Larger is stricter. |
| `local_depth_tolerance_m` | real .004, >0 m | Neighbour Z agreement tolerance. Larger tolerates slopes/noise; smaller can reject smooth but steep bands. This is camera-Z even with intrinsics. |
| `min_region_area` | int 25, 1..1000000 pixels | Minimum connected positive area. Larger suppresses small patches; cannot rejoin a band interrupted by missing depth. |

## Tuning and debugging

First check units, alignment, object mask and `excluded_mask`. Inspect actual
sensor noise before reducing metric thresholds. Then inspect per-region reasons
and the inward reference before changing detection strength.

- **No candidate, `insufficient_inward_support` / `insufficient_robust_inward_support`:**
  inspect missing depth and context. Increase context/cell cautiously or reduce
  band/gap if physically appropriate. A narrow object may have no independent
  inward baseline; reducing all support requirements does not solve this.
- **`inward_reference_not_planar`:** use smaller cells or a different context to
  avoid folds. Only raise `max_reference_error_m` when that variation is acceptable;
  the measured error is added to the height threshold.
- **Small smooth bands missing:** inspect positive `relative_height_m` and the
  effective threshold. Adjust `min_height_m` after measuring noise; if slope is
  being rejected, inspect `supported_mask` and `local_depth_tolerance_m`.
- **Noise candidates:** increase guard, agreement requirement, area or height.
  Coherent errors can still pass these tests.
- **Detached/internal ridge:** reduce `attachment_px`; see `rejections` with
  `not_attached_to_outer_edge`. Keep `attachment_px` greater than guard.
- **Wide band touches evaluation limit:** increase band/gap together with enough
  context, then verify baseline did not enter another feature. Full width is not
  measured by `inward_extent_px`.
- **Crop or unobserved boundary:** `unsafe_boundary_mask` and `excluded_mask`
  explain excluded pixels. There is no switch to assert a physical outer edge
  at a frame cut. Other observed edges can continue independently.

`candidate_mask`, `unknown_mask`, `evaluation_mask`, `reference_mask` (union of
accepted fit inputs), and `excluded_mask` are always returned. Exclusion is not a
negative measurement; `state='no_candidate'` only applies to the evaluated band.
`state='unknown'` means no accepted candidate and missing/ambiguous evaluation;
`state='candidate'` can coexist with unknown regions. Candidate representative
points are observed candidate pixels near the region centroid, not depth minima.
Candidates include area, bounding box, median/p95 height, inward extent,
`touches_band_limit`, and exact contributing fit `region_ids`. `regions` contain
per-fit status/reason, support, fit coefficients and reference residual.
`rejections` describe positive components rejected for size or attachment.

Enable `diagnostics=True` for outer/unsafe/valid/supported masks, outer distance,
per-pixel `source_region_ids`, representative points and individual context
indices. `result.reference_for_region(region_id)` returns a detached mask of
that fit's supported context inputs (including failed-fit inputs for diagnosis).
It requires diagnostics. The aggregate reference mask alone is not the baseline
for every candidate. Regions that fail fitting have no finite height result.

The heatmap uses fixed +/-`4*min_height_m`, red toward camera, magenta for missing
fit/result. Saturation does not give the actual height; use arrays/metrics.
Spatial agreement on one frame is not temporal stability.

Run the executable synthetic example to save metrics, arrays, PNG panels and a
selected fit's reference mask:

```sh
python examples/example_edge_bulge.py --output-dir /tmp/edge-bulge-demo
```

The destination must not exist. Synthetic examples do not validate camera-specific
depth accuracy or thresholds. Scene-specific evaluation requires current depth,
its scale, the object mask and sensor validity, not a photograph of a depth display.
