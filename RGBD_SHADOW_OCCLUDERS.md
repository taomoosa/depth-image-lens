# Dark shadows and nearer occluders with existing RGB-D APIs

For a fixed registered empty/current RGB-D pair, existing APIs can conservatively
select **visible raised target candidates within a declared camera-Z band**.
No new segmentation mode is required. Existing defaults remain unchanged.
Run `examples/example_rgbd_shadow_occluders.py` for a small ndarray example.

## Fixed settings across captures

The 5/80 mm example is not a universal preset. One fixed upper bound requires a
noise-separated gap between the largest legitimate target and the nearest unwanted
surface over the entire operating range. Object lifting, camera tilt and depth
reference drift can remove that gap. See [repeat-capture conditions and regression
counterexamples](RGBD_CAPTURE_STABILITY.md) before selecting deployment values.

## Which evidence to require

- `fusion_mode='depth_required'` selects only valid empty-current camera-Z
  differences greater than `depth_threshold_m`. A shadow whose depth equals the
  table therefore cannot pass, even if it is extremely dark or colored.
  This mode means depth only, not RGB AND depth.
- Exclude valid nearer surfaces with a caller-derived `work_roi`. Nearer surfaces can pass
  foreground selection; no mode semantically identifies an occluder.
- Use `missing_depth_policy='unknown'` for any color-assisted mode. The default
  `'color'` can select invalid-depth occluders from their RGB. The depth-required
  mode already requires valid depth regardless of this policy; spelling it out
  keeps the intended policy when switching modes.
- To preserve the earlier `color_seeded` restriction, intersect its mask with
  the depth-required mask, with the same ROI and missingness policy. Do not union
  it back in: strict chroma seeds may have valid depth equal to the table.

`color_seeded` and `evidence` alone are not dark-shadow rejection guarantees.
Even uniform RGB attenuation on a colored table can change encoded-Lab chroma
sufficiently to form a seed. The existing boundary detector's
[`uniform_color` shadow hypotheses](BOUNDARY_SHADOW.md) are useful only when their
brightness, color-model and seed assumptions hold. Very dark/clipped pixels do
not have reliable normalized RGB ratios. Lowering `shadow_min_ratio` alone does
not make these pixels safe, and object/shadow identical appearance is ambiguous.

## Declare the usable depth band

For millimetre inputs:

```python
from depth_image_lens import RGBDBackgroundSegmenter, background_difference

delta_mm, pair_valid = background_difference(
    depth_mm, empty_depth_mm, depth_unit='mm', background_unit='mm',
    invalid_values=(65535,))
near_excluded = pair_valid & (delta_mm > 80.)
roi = application_roi & ~near_excluded
result = RGBDBackgroundSegmenter(
    fusion_mode='depth_required', missing_depth_policy='unknown',
    depth_threshold_m=.005,
).detect(color, depth_mm, background_color=empty_color,
         background_depth=empty_depth_mm, color_order='RGB',
         registered_to_depth=True, depth_scale_m=.001,
         invalid_values=(65535,), work_roi=roi, diagnostics=True)
```

Here the admission band is `5 mm < empty-current Z <= 80 mm`. These are example
application bounds, not fitted defaults or physical height estimates. Choose a
maximum that contains the legitimate target and is separated from the unwanted
near object by a measured noise margin. Keep metric units explicit. Never subtract
unsigned arrays directly. Preserve original sensor masks and declared sentinels;
`0`, negative, nonfinite and declared sentinel samples remain unknown. If the
sensor provides extra validity masks, use them consistently both when deriving
the exclusion mask and in `detect`.

An absolute camera-Z near limit can instead use valid **current** depth, for
example `current_valid & (depth_m < declared_min_z_m)`. Build `current_valid` from
sensor validity, positive finite depth and sentinels, not color or a segmented
mask. Do not use returned `result.depth_valid_mask` as raw sensor validity: it is
already intersected with background-depth validity and the ROI. For tilted-camera
physical table height, use calibrated `CameraIntrinsics.deproject` and
`TablePlane.height` with a verified current table plane; camera-Z difference is
not table-normal height.

## Preserve unknown regions and diagnostics

Keep `near_excluded` and `~pair_valid` separately. Both are outside the downstream
observed target domain, but only the first has a usable measured near surface.
Missing depth does not establish what was present. Masking the near surface cannot
restore the target hidden behind it.

Pass `result.outline_observed_mask` into `OuterContourDetector.detect`. An outline
next to an observation/ROI cut is clipped or uncertain. Do not replace observation
with an all-true mask or fill the cut. For the optional color/depth intersection,
intersect both observation masks and exclude disagreement pixels; retain the
union of uncertainty masks plus disagreement separately for pick checks.

`outline.holes` describes binary mask topology only. An enclosed missing-depth
patch can appear there without per-hole visibility metadata; it is not a verified
physical hole. No hidden perimeter, full target area or safe grasp is inferred.

Useful arrays are the external exclusion/validity/ROI masks and the segmenter's
`depth_difference_m`, `depth_threshold_margin_mask`, `object_mask`,
`outline_observed_mask` and `uncertain_mask`. No new diagnostics API is needed.

## Necessary tradeoffs and counterexamples

- A true target thinner than the lower threshold or with missing depth is lost
  or left unknown, even when color supports it. Weakening the depth requirement
  to recover it can also admit indistinguishable table shadows.
- A legitimate raised part beyond the declared maximum is also excluded.
- A second object inside the admitted band is not identifiable as unwanted from
  depth alone. Its RGB may also match the target. Use an independently justified
  ROI, semantic exclusion, tracking or another observation when needed.
- Coherent depth bias larger than the lower threshold can make a shadow look like
  a raised surface. Thresholding cannot distinguish identical RGB-D observations.
- Color intersection can suppress depth artifacts, but loses same-table-color or
  unseeded true objects. Depth-only remains available for these cases.
- Registration errors, mixed boundary depths, sensor noise and wrong reference
  depth can still corrupt a silhouette. These synthetic regressions establish
  contracts and explicit failure cases, not real-image accuracy. Validate actual
  registered pairs before choosing deployment thresholds.
