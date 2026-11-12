# Connected color-region depth support (experimental)

`RGBDBackgroundSegmenter(fusion_mode="color_region")` tests an entire connected
color region, rather than requiring square-neighborhood support at each pixel.
It is opt-in. Existing modes and defaults are unchanged.

## Algorithm and evidence

1. Form background-difference candidates with full encoded Lab, including L.
   The combined background score must exceed `region_color_relaxation_ratio ** 2`.
2. Group current full-Lab colors around deterministic fixed representatives:
   choose the most frequent remaining color and assign colors within
   `region_color_tolerance`. A chain of small color steps cannot grow a group
   indefinitely. These groups can split a smooth gradient and are not semantic
   objects. Independently label each group's 8-connected components. Separate
   islands with identical colors never share depth support.
3. For each component, calculate valid depth coverage, the mean signed depth
   difference clipped to `+/- region_depth_clip_ratio * depth_threshold_m`, and
   the fraction of valid pixels above `depth_threshold_m`. Accept the region
   only when **all** its area, coverage, mean, and density gates pass.
4. Select its valid-depth pixels. Pixels accepted only through the region vote
   are inferred/unknown, not observed foreground depth. Missing depth is never
   filled. Farther-depth observations contribute negative evidence to the region statistics
   and are never selected. Do not discard them before computing the mean: doing
   so would positively bias zero-mean noise. A farther-depth line does not split
   the color region for statistical support; identical colors on either side may
   still share evidence.

Depth difference is empty-background camera Z minus current camera Z, not
physical table-normal height. Clipping prevents a few enormous outliers from
controlling the mean; it does not remove coherent bias. The mean includes
negative and subthreshold observations, not only foreground pixels. Density's
denominator is valid pixels, while coverage's denominator is all region pixels.

This mode deliberately trades local support and a growth-radius limit for whole
region support. It does not call the square-neighborhood `depth_support` helper.
`depth_support_*`, `strong_depth_ratio`, `supported_*`, `anchored_*`, and old
`color_growth_*` controls do not govern this mode. `missing_depth_policy` does not
permit filling absent depth here.

## Parameters and tuning

| Parameter | Default | Effect |
|---|---:|---|
| `region_color_tolerance` | 18 | Encoded Lab distance to a fixed representative. Larger values tolerate color variation but merge touching object/shadow colors. Not standard Delta E. |
| `region_color_relaxation_ratio` | 0.2 | Lower values admit weaker background differences, including gray; they also admit background drift. |
| `region_min_area` | 32 px | Minimum connected color-region area. Larger removes speckles but loses small, thin, or cropped pieces. |
| `region_min_valid_fraction` | 0.5 | Required fraction with valid depth. Missing observations do not vote background. |
| `region_mean_depth_ratio` | 1 | Required clipped mean divided by `depth_threshold_m`, strictly greater. Lower values admit shallow objects and systematic errors. |
| `region_min_depth_fraction` | 0.25 | Required fraction of valid pixels above `depth_threshold_m`. Higher rejects sparse support, including genuine noisy objects. |
| `region_depth_clip_ratio` | 3 | Clip each signed depth difference to this multiple of T before averaging. Smaller suppresses large outliers but also reduces genuine sparse tall-object support. Must exceed the mean ratio. |
| `region_max_color_groups` | 64 | Complexity cap (maximum 256). Unassigned candidates remain unknown; check the budget diagnostic. |

The usual `depth_threshold_m=0.005`, background Lab tolerances, ROI and validity
inputs still matter. `min_object_area` remains a final selection cleanup. First
inspect the region partition, then coverage, clipped mean, and density. Do not
lower every gate until a desired count appears. Tune on both real-object and
empty-table/shadow captures, at the intended resolution.

## Diagnostics and known failures

Use `diagnostics=True`. `region_label_map` identifies connected regions;
`region_color_group_map` identifies representative colors (not tracked IDs).
`region_statistics` gives each region's area, valid count/fraction, clipped mean
ratio, density, and accepted flag. It also appears in `stages['fusion']['regions']`
for strict JSON export without arrays. Per-pixel maps include
`region_depth_support_fraction`, `region_valid_fraction`, and
`region_clipped_mean_depth_m`. `region_color_budget_exceeded` reports incomplete
color grouping. The generic debug image includes categorical region decision
reasons; numeric maps/labels remain available directly.

**Important:** a touching shadow sufficiently close in color may belong to the
same region and borrow the object's depth support. Entire same-color holes can
also be filled as inferred pixels. This is a stronger failure than bounded local
growth: no distance limit protects the far end of that region. Increasing density
may reject the whole object instead of isolating its shadow. A consistent erroneous
depth offset can satisfy both mean and density. An integer label and the heuristic
`confident_foreground_mask` do not certify physical material or safe grasping.

If no color difference passes the candidate threshold, depth alone does not
create a region. Compare `depth_anchored` for that use case. Gradients, highlights,
and small color islands can fragment objects. Missing depth, rejected candidates,
and inferred regions retain uncertainty. Rejected color regions are not proof of
background. Raw depth and foreground masks are unchanged.

See `examples/example_rgbd_color_region.py` for a reproducible sparse-depth case.
Tests cover sparse support, noise, separate/touching shadows, holes, missing depth,
ROI/crop, outliers, parameter validation, and explicit failure cases. These are
synthetic checks, not real-camera accuracy or realtime performance claims.

## Illustrative synthetic comparison

Five random seeds for each of twelve small scenes were compared against four
existing modes (300 method/scene runs). No real sensor capture was used. A
40x40 gray region on gray background was at 6 mm, or at a mean 7 mm with
independent Gaussian depth-difference noise of standard deviation 5 mm.

- Noisy 7 mm: mean IoU 0.9901 (`color_region`), 0.9639 (`color_supported`),
  0.6538 (`depth_required`).
- Constant 6 mm: IoU 1.0 versus 0.0 for `color_supported`'s stronger seed gate.
- A same-color touching 40x40 shadow beside the object was entirely admitted:
  IoU 0.5 versus 0.9091 for bounded `color_supported`.
- Missing depth on half the object remained unfilled (IoU 0.5).
- A same-color 7x7 hole was inferred as foreground, while a background-colored
  hole is excluded by the color candidate domain.

These are demonstrations of structural behavior, not general accuracy limits.
The favorable noise model is independent and zero-mean; correlated sensor error
and color/depth registration error can invalidate the support test.

A further 240 parameter/scene/seed runs swept mean ratios 0.5/1/1.5 and
support densities 0.1/0.25/0.5/0.75, with 5 mm independent noise and T=5 mm.
At density 0.25, lowering the mean ratio from 1 to 0.5 recovered 94.6% of a
3 mm object, but also admitted 96.4% of a shadow with a systematic 4 mm
positive depth error. Zero-mean shadows remained excluded. Raising density to
0.75 lost the noisy 7 mm object; raising the mean ratio to 1.5 did too.
Thus lower the mean gate only after checking empty-table bias; increasing the
density gate is useful for sparse spurious support, not a substitute for bias
calibration. These percentages describe these synthetic scenes only.
