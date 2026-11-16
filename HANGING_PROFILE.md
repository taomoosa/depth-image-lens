# Hanging shape profiles

`HangingProfileAnalyzer` extracts **image-space measurements**, not a learned or
validated classification of clean hanging, snagging, grasp, or hidden folds.
A low hem alone does not prove clean hanging. A sleeve, normal U shape or a
single dangling tail can have a similar extreme extent.

## Basic use

```python
from depth_image_lens import HangingProfileAnalyzer

analyzer = HangingProfileAnalyzer(
    min_component_area=10,
    min_vertical_run_px=3,
    smoothing_window_px=5,
    quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
    reach_drop_px=50,
    upper_band_height_px=25,
)
result = analyzer.analyze_mask(
    object_mask,                 # bool H x W, selected hanging object
    reference_y=30,               # known gripper lower edge, image y downward
    expected_x_range=(80, 560),   # optional fixed reference span, x1 exclusive
    observed_mask=observed,       # observable pixels, including background
    uncertain_mask=inferred,     # repaired or uncertain pixels are not evidence
    mask_complete=False,         # false mask pixels are not guaranteed absence
    diagnostics=True,
)
print(result.metrics['drop_quantiles_px'])
print(result.metrics['reaching_width_fraction'])
bottom = result.profiles['bottom_y_px']
row_width = result.profiles['row_width_px']
rgb_debug = result.render_debug()  # requires optional OpenCV [debug]
```

Inputs are aligned, in the same cropped coordinate system. `reference_y` is a
finite scalar; it can be above the image (negative). Rows above it are excluded.
No depth values are required here: obtain the mask and observation validity from
the preceding depth/segmentation stage. Image y is **not** metric vertical height
in an oblique camera. Resizing changes pixel thresholds. Camera geometry,
gripper spacing and object size should remain consistent or be calibrated.

`expected_x_range` defaults to the whole image, **not the detected object width**.
Use a known gripper/reference span to prevent a single narrow dangling strip from
scoring 100% reach. Do not repeatedly adapt this denominator to the measured
strip. Nothing automatically identifies the grippers or removes them: choose a
reference below the grippers and exclude their pixels in the supplied mask.

## Outputs and interpretation

All column arrays have original image width W, and all row arrays height H.
Columns outside the reference span, or without a supported foreground run, use
NaN. Missing columns are never filled with zero or interpolated.

- `raw_bottom_y_px`: last observed, non-uncertain foreground pixel in each column,
  before small-component/run rejection.
- `bottom_y_px`: last pixel of a vertical run at least `min_vertical_run_px` long
  in components of at least `min_component_area` pixels (8-connectivity).
- `drop_px`: `bottom_y_px - reference_y`. These are visible extents/lower bounds,
  not assured physical cloth lengths.
- `smoothed_bottom_y_px`, `smoothed_drop_px`: local median for display/inspection;
  smoothing never bridges missing columns. These estimates are not observations
  and can be above or below a column's observed extent. **Metrics use the
  unsmoothed supported profile**, not these estimates.
- `row_width_px`: number of retained observed foreground pixels at each y. It is
  occupancy, not bounding-box width. Gaps are not counted.
- `row_width_fraction`: row occupancy divided by the reference span width.
- `column_observed_fraction`: observed and non-uncertain fraction of the
  below-reference measurement domain, not an object detection confidence.
- `bottom_censored`: true when a supported bottom is only a lower bound: mask
  completeness is not attested, it touches the bottom frame, or unknown/inferred
  pixels lie below it in the same column, or filtering removed observed
  foreground below the supported bottom.
- `exact_drop_px`: only non-censored supported columns. "Exact" means within
  the supplied mask/observation assumptions, not ground-truth physical accuracy.

Metrics:

- `drop_quantiles_px`: configurable quantiles (`p10`, `p25`, `p50`, `p75`, `p90`
  by default) across columns with supported foreground. Percentiles interpolate
  as NumPy's default linear quantile. Every column has equal weight.
- `exact_drop_quantiles_px`: same, restricted to non-censored columns. This
  subset can be biased: report its sample count rather than treating it as the
  full object's distribution.
- `drop_mean_px`, `drop_iqr_px`, `drop_mad_px`: mean, Q75-Q25, and median absolute
  deviation of the supported (not smoothed) extents. Null if no samples.
- `reaching_width_fraction`: supported columns reaching `reach_drop_px` divided
  by the whole reference span. It is a visible-evidence ratio, not a binary
  clean-hanging decision; absent/unknown columns do not contribute.
- `reaching_observed_columns_fraction`: the alternative denominator (supported
  foreground columns only), useful diagnostically but unsafe alone for tails.
- `upper_area_fraction`: retained observed pixels within
  `reference_y <= y < reference_y + upper_band_height_px`, divided by all retained
  observed foreground pixels below the reference. Missing pixels can bias it.
- component counts, observation fraction, removed support, and boundary flags.

Combining lower quantiles/reaching width and upper-area concentration helps
separate "one long tail, most material bunched near the gripper" from broad
hanging material. Neither median nor quantiles alone are universally sufficient:
an equal bimodal mixture has a particularly unstable median. Use the profile,
IQR and width/area measurements together and calibrate on actual objects.

## Observation and fragment handling

`observed_mask` marks known pixels, not just foreground. If omitted, all pixels
are assumed observed and the result explicitly warns about that assumption.
Pass depth validity or the relevant observation map; zero depth is not a short
object. `uncertain_mask` may cover repairs, occlusion, or uncertain membership.
Those pixels do not contribute observed area or supported profiles.

Small components are discarded but surviving fragments are **not** joined,
assigned to physical objects, or reduced to the largest component. Use each
selected component separately, or explicitly union fragments known to belong to
one object. A disconnected component may be real cloth or noise. Vertical run
filtering suppresses isolated bottom pixels but may remove very thin real
features; all removed pixels are visible in diagnostics.

`mask_complete=True` is a caller assertion that false mask pixels mean absence.
It must not be used to hide detection uncertainty. Without it all supported
bottoms are lower bounds. Unknown columns, crop and boundary touches are reported
rather than classified as snagging. `complete_bottom_distribution` concerns
supported-column bottoms only; it does not prove the object is fully visible
horizontally or that segmentation is correct.

## Using existing detector components

```python
labels = grasp_result.diagnostics['component_labels']  # diagnostics=True upstream
for component in grasp_result.metrics['components']:
    mask = labels == component['id']
    profile = analyzer.analyze_mask(
        mask, reference_y=30, expected_x_range=(80, 560),
        observed_mask=grasp_result.diagnostics['valid_reference_mask'],
        mask_complete=False,
    )
```

Do not count fragmented depth components as necessarily different objects. A
known object's union mask is often a better analysis unit. For `BridgeDrapeDetector`,
its left/right drape masks may also be analyzed; its `sag_path_xy` is an upper
outline, not this lower profile.

## Debug and serialization

Use `diagnostics=True` to retain input, measurement domain, observed/retained
mask, supported-run mask, removed small components, uncertain foreground and
missing observations. `render_debug(color=rgb)` returns RGB with raw (orange),
supported (cyan), and smoothed (yellow) bottom profiles, censored markers (pink),
a reference line, and a row-width chart. It requires OpenCV only for rendering.

`result.to_dict()` returns JSON-safe metadata/metrics without image arrays.
`result.to_dict(include_profiles=True)` includes 1D arrays, converting NaN to
JSON null. No clean/snaggled state is inferred; `classification='not_performed'`.
