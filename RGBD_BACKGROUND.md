# Current and empty-scene RGB-D segmentation

For all parameter ranges, direction of adjustment, symptom-based troubleshooting,
and a runnable NPZ/debug workflow, see [the tuning guide](RGBD_PICK_TUNING.md).

`RGBDBackgroundSegmenter` returns a bool `object_mask` from current color/depth
and empty-scene color/depth pairs. It uses the existing boundary detector's
encoded-Lab background comparison, then fuses signed depth evidence. Existing
classes and their defaults are unchanged. Requires the optional `boundary` or
`outline` OpenCV extra. Detection itself writes no files.

```python
from depth_image_lens import RGBDBackgroundSegmenter, OuterContourDetector

seg = RGBDBackgroundSegmenter(
    background_chroma_tolerance=20.,
    background_luma_tolerance=60.,
    depth_threshold_m=.005,
    depth_margin_m=.001,
    fusion_mode='union',
    missing_depth_policy='color',
).detect(
    current_color, current_depth,
    background_color=empty_color, background_depth=empty_depth,
    color_order='RGB', registered_to_depth=True,
    depth_scale_m=.001, diagnostics=True,
)
outline = OuterContourDetector().detect(
    seg.object_mask, observed_mask=seg.observed_mask,
)
# For the existing internal color/depth boundary pipeline:
# boundary = OverheadBoundaryDetector().detect(
#     current_color, current_depth, color_order='RGB',
#     registered_to_depth=True, depth_scale_m=.001, object_mask=seg.object_mask)
seg.save_debug('rgbd-background.png', current_color, color_order='RGB')
```

Run `PYTHONPATH=src python examples/example_rgbd_background.py` from the checkout.
Add `--output /existing/directory/new-file.png` for an explicit debug image.
The generated scene has two components and one hole. This checks API behavior;
it is not a real-image accuracy experiment.

## Input contract

- Both colors are uint8 HxWx3, both depths numeric HxW, all on exactly the same
  pixel grid. Declare `color_order='RGB'` or `'BGR'` for both colors.
- `registered_to_depth=True` explicitly declares that the two modalities and
  the two captures are already aligned. Keep the camera, intrinsics, resolution,
  crop, background geometry and pose fixed. A static but sloped background is
  supported by per-pixel subtraction; no plane calibration is required.
- Supply optical-axis Z, not range along a ray. `depth_scale_m` is required and
  applies to both depth arrays, including floats: use 1 for metres, .001 for mm.
  The detector never infers units from dtype and never aligns/resizes inputs.
- Nonpositive, nonfinite or `invalid_values` sentinel samples in either depth
  are missing. Sentinels use original count units. Overflow/underflow in metric
  conversion is also unavailable depth. Subtraction uses float64, avoiding
  unsigned wraparound. Invalid difference pixels are NaN, not zero.
- Optional bool HxW `color_valid_mask`, `background_color_valid_mask`,
  `depth_valid_mask`, `background_depth_valid_mask` declare real samples in each
  capture. Mark inpainted/occluded/unreliable data False. Color validity cannot be
  inferred from a black pixel. `work_roi` restricts the selection domain; outside
  it `object_mask`, `observed_mask`, and `uncertain_mask` are False (excluded).
- `background_check_mask` declares known unchanged background in current pixel
  coordinates; provide at least 25 color-valid pixels when supplied or when
  compensation is enabled. It may lie outside the selection ROI. The detector
  does not verify that the reference is empty or that this region is background.
- Invalid contracts raise `ValueError`; they never trigger border-based guesses.

## Evidence and fusion

Color: using OpenCV encoded 8-bit Lab, let `dC = hypot(da, db)` and `dL = abs(dL)`.
A pixel is color foreground if `(dC/Tc)^2 + (dL/Tl)^2 > 1`. These units are not
standardized Delta E. A larger luma tolerance reduces brightness/shadow sensitivity
but also misses brightness-only objects. Existing saturation/clipping diagnostics
are retained. `background_score_margin` marks proximity of the ellipse radius to
1 as uncertain; it does not change selection.

Depth: `dZ = empty_depth_m - current_depth_m`. Positive means closer to the
camera, not height normal to a table. Depth foreground requires `dZ > T`.
A change with `dZ < -T` is farther evidence. Changes within the threshold are
unresolved depth differences, not proof of background. Exact thresholds use
floating-point comparisons; values within `depth_margin_m` of either signed
threshold are flagged uncertain, whether selected or not.

| Setting | Selection behavior |
| --- | --- |
| `fusion_mode='union'` (default) | Depth foreground OR color foreground, with a farther-depth veto on the color contribution. Same-color objects can be found from depth; thin objects can be found from color. |
| `fusion_mode='depth_required'` | Only depth foreground. Reduces shadow-driven selection, but misses objects below the depth threshold or with missing depth. |
| `missing_depth_policy='color'` (default) | In union mode, retain color-only candidates where either depth is missing, marked uncertain. |
| `missing_depth_policy='unknown'` | Require valid depth even for color candidates; missing depth is unobserved. In depth_required mode, depth is required regardless of this setting. |

Color-only pixels remain uncertain even when depth is valid. Farther changes are
also uncertain; `conflict_mask` records color foreground at farther-depth pixels.
Valid-depth foreground remains selectable when color is missing or rejected.
Missing either modality and clipped/near-threshold evidence are marked uncertain.
`uncertain_mask` is independent of `object_mask`: uncertainty does not automatically
remove a candidate. Unselected pixels without uncertainty merely lack a detected
change under the declared model; they are not certified background.

No closing, erosion, polygon filling or inferred depth filling is applied.
Nonconvex shapes, holes, small thin features and multiple components are retained
as measured. Optional `min_object_area` removes 8-connected components smaller
than that pixel count and marks the discarded pixels uncertain. Default 1 removes
nothing. Diagonally touching objects still form one component.

## Settings and debug

| Constructor setting | Default | Meaning |
| --- | --- | --- |
| `background_chroma_tolerance` | 20 | Positive encoded-Lab chroma scale Tc |
| `background_luma_tolerance` | 60 | Positive encoded-Lab luma scale Tl |
| `background_score_margin` | .2 | Ellipse radius uncertainty margin, 0..1 |
| `depth_threshold_m` | .005 | Positive camera-Z change threshold T in metres |
| `depth_margin_m` | .001 | Nonnegative uncertainty margin, strictly less than T |
| `min_object_area` | 1 | Minimum retained 8-connected area, pixels |
| `background_compensate` | False | Apply median Lab offset from declared unchanged check region |
| `background_check_p95` | 20 | Maximum corrected check residual, unweighted encoded-Lab distance |
| `background_max_offset` | 40 | Maximum permitted Lab compensation vector norm |

Check/compensation reuse the existing color-reference implementation. A rejected
check disables color evidence and keeps usable depth. It never falls back to
border colors. Too few check pixels is an input error. These checks concern color
only; depth reference noise/drift is not compensated or validated by that region.

Result masks include `object_mask`, `uncertain_mask`, `observed_mask`,
`color_foreground_mask`, `depth_foreground_mask`, `depth_valid_mask`, and
`conflict_mask`. `observed_mask` describes availability under the fusion policy,
not semantic certainty and not raw current-depth validity alone. Pass it to
`OuterContourDetector` so unavailable pixels and ROI cuts are reported as such.
Do not use `~uncertain_mask` as the outline observation domain by default: that
would cut out useful color-only objects and create artificial contours. Propagate
segmentation uncertainty separately when interpreting outlines. For metric outline
lengths, supply current-depth validity and camera intrinsics under that detector's
normal contract; do not substitute the pair-validity mask without considering it.

The fusion stage reports `candidate` when selected pixels exist,
`no_change_candidate` when observations exist but none pass selection,
`insufficient` when the ROI has no usable observations under the policy, and
`empty_domain` when the ROI itself is empty.

`diagnostics=True` retains numeric `depth_difference_m`,
`background_chroma_distance`, `background_luma_distance`,
`background_combined_score` (threshold 1), plus validity, farther, threshold-margin,
color-only and filtered masks. The color maps describe the comparison even if the
color reference check rejects it; consult `stages['color_reference']` and
`color_usable_mask` before treating them as usable. `to_dict()` contains JSON-safe
parameters, stage reasons and counts; numeric arrays remain in `diagnostics`.
`render_debug`/`save_debug` use separate overlays and optional evidence heatmaps.

## Limits and tuning

Defaults are starting values, not calibrated sensor thresholds. Estimate depth
noise from repeated empty captures and place T above that noise; lower T admits
thinner surfaces but also noise and registration edge artifacts. Tune Tc/Tl using
actual lighting and materials. Optional illumination compensation is a global
Lab offset, not spatial shadow removal, exposure calibration or camera tracking.

Strong shadows, highlights, auto-exposure/white-balance changes and moving
backgrounds can look like color-only objects. The depth_required mode trades
these false positives for missing thin/missing-depth objects. A same-color object
with unresolved or missing depth is indistinguishable from the background. A
same-color, same-depth object cannot be detected by either modality. Reference
contamination, transparent/reflective surfaces, mixed depth edges, camera motion,
rolling shutter and mismatched alignment can create false contours. Nearby or
touching objects are not split semantically. No learned object recognition,
subpixel boundary recovery, camera operation, or real-image precision claim is
provided. Synthetic tests establish only these API/evidence contracts.
