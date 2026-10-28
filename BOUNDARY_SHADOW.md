# Optional uniform-color shadow hypotheses

This opt-in mode is for a single approximately uniform deformable object on an
approximately uniform table, with a separately captured empty reference. It is
not semantic segmentation or guaranteed shadow invariance. Different RGB values
alone are insufficient: a dark gray object and a gray table shadow can be
identical. The default `background_shadow_mode="off"` preserves the existing Lab
background-difference selection.

```python
import numpy as np
from depth_image_lens import OverheadBoundaryDetector

# current and empty: aligned uint8 HxWx3 RGB, same camera/crop.
# Locations below are illustrative: inspect both patches on your own scene.
check = np.zeros(current.shape[:2], bool)
check[10:30, 10:30] = True   # unchanged visible table in BOTH images
seed = np.zeros_like(check)
seed[80:100, 100:120] = True # one small representative object-color patch
model = OverheadBoundaryDetector(background_shadow_mode="uniform_color")
result = model.detect(current, color_order="RGB", background_color=empty,
                      background_check_mask=check, object_color_seed=seed,
                      diagnostics=True)
print(result.stages["shadow_separation"])
mask = result.object_mask
uncertain = result.uncertain_mask  # can overlap foreground AND suppressed pixels
result.save_debug("shadow-debug.png", current)  # refuses an existing file
```

Registered optical-axis depth remains optional and is not used to decide color
separation. Missing or unusable optional depth does not stop the color stages.
An explicit `object_mask` takes priority and reports shadow separation as ignored.
The seed is a strict bool HxW sample, not a complete object mask. It is ignored when
the new mode is off. A missing or malformed seed leaves foreground evidence in
place and reports insufficient shadow separation, rather than aborting detection.

## Method and safeguards

The existing reference alignment, validity and check-patch validation run first.
Optional `background_compensate=True` still estimates one global additive encoded
Lab offset; this is not multiplicative exposure calibration. The corrected reference
is then converted to RGB for this mode. A valid check patch does not certify the
whole reference is empty or correctly registered.

For usable pixels, RGB is divided by R+G+B. Euclidean distance between these
normalized **encoded** RGB vectors measures color-ratio similarity. Mean encoded
RGB current/reference intensity gives a brightness ratio. This simple ratio model
resembles the darker-background shadow assumption documented for
[OpenCV MOG2](https://docs.opencv.org/4.13.0/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html),
but this implementation uses an explicit image pair and samples, not a learned
video background model or copied MOG2 implementation.

Suppression requires a declared stable table check and a distinguishable object
color sample. Reference and seed chromaticity p95 spreads must each fit the chosen
tolerance; their median distance must exceed twice the tolerance plus both spreads.
At least 25 usable seed pixels and 90% seed coverage are required; the reference
requires 25 usable pixels and 80% usable coverage. Seed/check overlap is rejected.
These fixed support guards are not performance guarantees or semantic verification.
Keep clutter outside the image/domain or mark genuinely unusable reference pixels
with `background_valid_mask`; never hide target pixels just to improve a score.

Only existing foreground pixels that resemble the reference chromaticity, satisfy
the brightness-ratio bounds, and are farther from the object model are suppressed.
No foreground is invented or filled. Suppressed pixels remain in `uncertain_mask`:
this is a shadow hypothesis, not confirmed empty table. Dark/clipped or poorly
modeled retained pixels also remain uncertain. Insufficient model separation keeps
legacy foreground candidates and marks them uncertain. It cannot recover object
pixels already indistinguishable from background in the original difference.

## Parameters and tuning order

| Parameter | Default | Units / effect |
|---|---:|---|
| `background_shadow_mode` | `"off"` | Enable explicitly with `"uniform_color"`. |
| `shadow_chroma_threshold` | 0.04 | Euclidean normalized encoded RGB distance; (0,0.2]. Higher allows more color variation, but also requires wider model separation. |
| `shadow_intensity_floor` | 20 | Mean encoded RGB on 0–255; [1,254). Below this, color ratios are unreliable and suppression is disabled locally. |
| `shadow_min_ratio` | 0.35 | Minimum current/reference mean brightness; (0,1]. Lower permits deeper shadow hypotheses and increases dark-object risk. |
| `shadow_max_ratio` | 1.15 | Maximum ratio; [1,2]. Allows modest brighter changes with similar color; lowering toward 1 restricts this. |
| `color_distance` | 30 | Existing encoded-Lab foreground threshold, not standardized Delta E. |

All defaults are provisional. First verify reference registration and inspect the
table and object sample patches. Check `status`, reasons, model separation and
spread before adjusting tolerances. Then inspect foreground retention, shadow
false positives and uncertainty separately. Adjust the brightness-ratio interval
only for measured lighting variation, and the chromaticity tolerance only after
checking noise/color diversity. Do not remove uncertainty from performance counts.
For brightness-only, near-color, patterned or colored-illumination cases, use manual
mask/ROI refinement or acquire more informative inputs; do not force suppression.

`stages["shadow_separation"]` records thresholds, support, model separation/spread,
candidate and suppressed counts, reasons and warnings. With `diagnostics=True`,
HxW arrays include `shadow_candidate_mask`, `shadow_suppressed_mask`,
`shadow_unreliable_mask`, `object_color_seed_mask`, `shadow_chroma_distance` and
`shadow_intensity_ratio`. Numeric unavailable values are NaN in arrays, not zero;
`to_dict()` remains strict-JSON safe. Render/save debug includes these heatmaps.

## Evidence limits

Controlled synthetic cases can measure retention and shadow false positives.
Existing real samples have no verified independent empty-scene pairs; reconstructed
backgrounds using target masks leak target information, and digital shadows are
perturbations of the same source, not new real observations. These tests do not
establish field segmentation accuracy. Capture independent empty/current pairs
under the actual camera and lighting before deployment. Printed surfaces, colored
shadows, saturation, deep darkness, changed camera pose and contaminated samples
violate or weaken the assumptions. A single uniform sample does not identify every
object pixel or distinguish a second dark object from shadow.
