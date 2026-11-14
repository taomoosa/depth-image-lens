# Local color-guided RGB-D segmentation

`RGBDBackgroundSegmenter(fusion_mode="color_local")` is an opt-in alternative
for weak color contrast and spatially scattered depth support. Existing modes
and their defaults are unchanged. Supply registered current color/depth and
empty-scene color/depth on the same pixel grid.

```python
segmenter = RGBDBackgroundSegmenter(
    fusion_mode="color_local",
    local_support_radius_px=4,
    local_mean_depth_m=0.0015,
    local_min_depth_fraction=0.18,
    local_growth_radius_px=2,
    local_max_reference_separation_m=0.10,  # optional camera-Z band
)
result = segmenter.detect(
    color, depth, background_color=background_color,
    background_depth=background_depth, color_order="RGB",
    registered_to_depth=True, depth_scale_m=0.001,  # for millimetre arrays
    diagnostics=True,
)
mask = result.object_mask
```

Use `depth_scale_m=1.0` for metre arrays. The upper band is **camera-Z difference**,
not height normal to the table, and must cover the allowed object geometry.
It is disabled by default (`None`). It excludes high measured pixels from both
statistics and growth, but cannot guarantee removal of all arm-boundary residue.
`work_roi` can exclude known irrelevant regions. Invalid depth is never filled.

## Algorithm

1. Build a relaxed color-change candidate mask, including luminance differences.
2. At each candidate pixel, aggregate signed depth difference in overlapping
   neighborhoods, weighting samples by full-Lab similarity to the center color.
   Negative differences remain negative; extreme values are clipped before
   averaging. There are no non-overlapping square output blocks.
3. Require both a positive mean and sufficient above-threshold depth density,
   valid coverage, and at least three effective weighted observations. Remove
   tiny seed components.
4. Separately identify locally table-like pixels. Grow only a bounded distance
   from accepted seeds, using a fixed source-seed color and local color edges.
   Growth never crosses table-like, invalid, farther-depth or upper-band pixels.

Unlike `color_region`, support does not authorize an entire connected region.
Unlike `color_seeded`, seeds do not require chroma alone. This is still a spatial
hypothesis, not a calibrated probability, semantic material boundary, or grasp
safety decision.

## Tuning

Start with the noise measured in repeated **real empty-scene frames**. The
example defaults are starting values, not universal camera calibration.

| Parameter | Default | Meaning / tradeoff |
|---|---:|---|
| `depth_threshold_m` | 0.005 | Per-sample foreground threshold used by density. Lower for very shallow noiseless objects, but table noise exceeds it more often. |
| `local_support_radius_px` | 4 | Overlapping `(2r+1)^2` neighborhood. Larger reduces independent noise but mixes boundaries and costs more. Allowed 1..10. |
| `local_mean_depth_m` | 0.0015 | Required clipped signed local mean. Lower recovers thin objects but admits bias. |
| `local_min_depth_fraction` | 0.18 | Required weighted fraction above `depth_threshold_m`. Compare with the measured empty-table exceedance rate. |
| `local_color_sigma` | 12 | Encoded-Lab scale for statistics weights. Smaller protects color edges, but color variation reduces effective support. |
| `local_growth_radius_px` | 2 | Maximum 8-neighbor boundary growth. Reduce to 0 or 1 to limit shadow leakage; missing edges may remain. |
| `local_color_tolerance` | 18 | Maximum full-Lab distance to source seed and preceding pixel during growth. |
| `local_color_relaxation_ratio` | 0.2 | Candidate background score must exceed its square. Larger rejects weak differences, including thin gray objects. |
| `local_min_valid_fraction` | 0.6 | Weighted valid fraction of candidate neighbors. Missing pixels never become object pixels. |
| `local_background_depth_m` | 0.00075 | Upper local mean for a table-like growth barrier. Must be below the foreground mean. |
| `local_background_max_depth_fraction` | 0.15 | Maximum density for a table-like barrier. Must be below foreground density. |
| `local_depth_clip_m` | 0.015 | Signed clipping magnitude for statistics, not a maximum object height. |
| `local_seed_min_area` | 8 | Minimum connected local-seed area; increasing removes small objects too. |
| `local_max_reference_separation_m` | None | Optional upper band described above. |

The background and foreground gates deliberately leave an unresolved interval.
A table-like classification can reject a genuinely very thin object. Lowering
the foreground mean alone cannot recover an object if no depth samples exceed
`depth_threshold_m`. Zero depth evidence plus identical appearance is not enough
to distinguish a thin object from a shadow.

## Diagnostics and provenance

- `local_candidate_mask`, `raw_local_seed_mask`, `local_seed_mask`
- `local_mean_depth_m`, `local_depth_support_fraction`, `local_valid_fraction`,
  `local_effective_samples`
- `local_background_mask`, `local_upper_excluded_mask`, `growth_step`
- `result.inferred_foreground_mask`: includes subthreshold center pixels accepted
  by aggregate support and all boundary growth; these remain `unknown_mask=True`.
- `result.confident_foreground_mask`: local seeds with center depth support and
  reliable reference color. Confidence is heuristic, not physical certainty.
- `result.confident_background_mask`: locally table-like under these settings,
  **not proof that no thin object exists**.

Debug decisions: 1 local seed, 3 table-like, 4 unselected raw depth, 5 unresolved,
6 area-filtered, 7 inferred foreground. Invalid depth and high-band exclusions
remain unknown. Mask repair is separate and must not be represented as measured
depth support.

## Reproducible synthetic evaluation

Run `PYTHONPATH=src python examples/example_rgbd_color_local.py`. It evaluates
10 fixed seeds per case on 60x100 arrays: 3 mm shallow objects (20 mm in the thick
case), independent Gaussian depth noise with sigma 4 mm, touching table shadows,
and gray/color-varied objects. RGB background is 140, weak object 125, other
objects and shadows 80; the mottled object has RGB variation sigma 8. No real
camera images are included or claimed.

Measured mean object recall / shadow acceptance:

| Case | color_region | color_supported | depth_anchored | color_local |
|---|---|---|---|---|
| Weak object color, different shadow color | 0 / 0 | .024 / 0 | .003 / .001 | .976 / .005 |
| Same-color touching shadow | 0 / 0 | .024 / 0 | .010 / .001 | .976 / .068 |
| Mottled object | 0 / 0 | .039 / .001 | .346 / 0 | .974 / .064 |
| Thick object, same-color shadow | 1 / .901 | 1 / .122 | .996 / .003 | 1 / .129 |
| Shadow with coherent +3 mm depth bias | 0 / 0 | .030 / .034 | .010 / .024 | .976 / .979 |

All modes use their defaults, including the same 5 mm raw depth threshold. This
compares specific operating points, not each mode's optimally tuned performance.
The bias counterexample is intentional: larger neighborhoods do **not** remove
correlated error. Thick same-color objects also leak some support at their border.
This mode is not always better than `depth_anchored`.

On the test machine (Intel Core i7-14700), a 640x480 call with diagnostics enabled,
one warm-up and three timed runs took a median 0.73 s; capture and rendering are
excluded. Cost is O(HW * radius^2) with bounded working arrays. Measure on the
target camera/CPU, and avoid raising the radius without checking latency.

There is no automatic empty-table noise calibration in this mode. Foreground
thresholds are explicit. Colors are individual fixed source-seed colors, not an
estimated single global material color; different seed colors can represent
shading within the object. Automatic noise estimation without a caller-declared
unchanged region could incorrectly learn a thin object as background.
