# Color-first weak-depth evidence

`RGBDBackgroundSegmenter(fusion_mode="color_first")` is an **opt-in** alternative
for thin objects with clear color boundaries but no pixels exceeding the usual
foreground-depth threshold. Existing modes, including `color_core` training,
remain unchanged. This is experimental single-frame change segmentation, not
material identification or a calibrated statistical test.

## Method

1. Compare current color against the registered empty-table color image. Form
   relaxed color-difference candidates, then partition them by fixed Lab color
   prototypes and 8-connected components. Depth does not seed these regions.
2. Erode each region **only for measurements**. The candidate outline is retained.
   Read all valid signed `background_depth - depth` differences in that interior,
   including zero and negative values. Clip outliers symmetrically.
3. Form a surrounding ring, excluding **all** changed-color candidates, invalid
   color and out-of-ROI pixels. Compare the region with this local table reference,
   rather than treating a global positive depth bias as object evidence.
4. Require sufficient valid coverage and spatial blocks on both sides. Compute
   pixel mean/median/10th/90th percentiles and block means. Both the region's
   mean and median must exceed the reference's by the required separation, and
   enough interior blocks must agree. Required separation is
   `first_min_depth_delta_m + first_noise_multiplier * Q90(abs(reference_block_mean - median(reference_block_mean)))`.
   The spread is **not divided by sqrt(pixel count)**. Larger regions do not
   manufacture high confidence from spatially correlated errors.
5. Accept the color region only if all gates pass. Invalid/too-high depth pixels
   are never filled. Rejected or insufficient candidates remain unknown, not
   proven background. Selected pixels without raw depth foreground support are
   uncertain inference (`inferred_foreground_mask`), not measured surface support.

No requirement for raw foreground-depth pixels or strong training samples exists
in this mode. `depth_threshold_m` still controls the raw foreground diagnostic
and inferred-versus-observed provenance, not the region acceptance threshold.

## Minimal use

See [`examples/example_rgbd_color_first.py`](examples/example_rgbd_color_first.py).
Use registered, same-sized current and background RGB/depth arrays. Integer
millimeter depth is supported using `depth_scale_m=0.001`. Submillimeter means
can arise by averaging integer measurements; this does not create submillimeter
sensor accuracy.

## Adjustment order

| Parameter | Default | Purpose / tradeoff |
|---|---:|---|
| `first_color_relaxation_ratio` | 0.2 | Candidate score exceeds this squared; lower admits weaker color changes but also shadows/noisy table and can remove the table reference ring. |
| `model_luma_tolerance`, `model_chroma_tolerance` | 18, 18 | Fixed color grouping, independently brightness/chroma; larger groups can merge touching shadows or objects. These are encoded OpenCV Lab units, not standard Delta E. |
| `model_training_inset_px` | 2 | Interior-only erosion to reduce mixed edge pixels. Reducing helps narrow shapes but increases depth-edge contamination. |
| `first_min_depth_delta_m` | 0.00075 | Minimum **relative** mean and median separation from surrounding table. Lower helps thinner objects but admits local depth bias. |
| `first_noise_multiplier` | 2.5 | Extra margin from surrounding block spread. Lower increases sensitivity and false positives under correlated noise. Not a Gaussian sigma or confidence level. |
| `first_min_block_fraction` | 0.75 | Required fraction of interior block means above the same relative threshold. |
| `first_block_size_px` | 8 | Spatial aggregation scale. Only blocks with at least half their pixel footprint in the region and adequate valid depth contribute. |
| `first_min_blocks` | 4 | Minimum usable interior blocks, and surrounding blocks when no surround override is set. |
| `first_min_surround_blocks` | None | Optional surrounding minimum only; `None` inherits `first_min_blocks`. An integer in 1–10000 sets the surrounding count separately. `0` bypasses the count check, not the need for measured surrounding depth. |
| `first_min_valid_fraction` | 0.6 | Required interior/reference coverage and within-block coverage. |
| `first_min_area` | 64 | Minimum candidate component area. |
| `first_ring_radius_px`, `first_ring_gap_px` | 16, 3 | Chebyshev ring extent and edge-contamination exclusion gap. Requires gap < radius. |

Also used: `background_luma_tolerance`, `background_chroma_tolerance`, reference
color compensation/validity settings, `local_depth_clip_m` (default 15 mm),
`local_max_reference_separation_m` (optional upper-camera-depth exclusion),
`region_max_color_groups` (candidate partition cap), and `min_object_area`.
`model_max_colors`, training policy, growth radius, background competition margin,
`model_reject_farther`, local seed/density thresholds are **not used** here.
Negative depth is retained in regional statistics instead of vetoing individual
pixels. Choose the upper-depth limit for the scene if an elevated arm is present.

### Sparse surrounding blocks

If only `first_failed_surround_blocks_mask` is set for a region, keep the
interior and depth settings unchanged and try `first_min_surround_blocks=3`
or `1`, depending on the observed block count. Use `0` only if you want
to evaluate a ring with no qualifying block but enough valid table depth, such
as a thin reference strip that straddles the block grid. The setting changes
only `color_first`.

When the relaxed count passes with fewer than two usable surrounding blocks,
the comparison uses the same valid, clipped signed ring pixels. Their median
replaces the median of block means for the interior-block comparison; the
90th percentile of their absolute deviations from that median replaces the
surrounding block spread. The configured noise multiplier, minimum separation,
pixel mean/median gates and interior support fraction still apply. This does
not assume zero noise just because there are zero or one usable blocks.

The unchanged valid-coverage gate still applies and the pixel fallback also
requires at least `max(4, ceil(max(4, block_size² // 2) * min_valid_fraction))`
valid pixels (20 with the defaults). An empty, invalid or too small measured
ring remains insufficient. More qualifying blocks give a better basis for a
local comparison; a sparse strip can be less representative, and a flat
measured spread does not prove the absence of residual depth bias.

### Optional table anchors for a frame-wide depth offset

Configure points on exposed table that normally remain outside the cloth, for
example two interior points in each upper corner. Coordinates are integer
`(x, y)` on the actual input array; in the tuner use the normalized **crop-local**
coordinates, not the full image. With four points, three agreeing votes are
required; if an entire two-point corner is covered, calibration normally skips
rather than choosing the remaining corner or the largest object.

```python
segmenter = RGBDBackgroundSegmenter(
    fusion_mode="color_first",
    first_table_xy=[(30, 25), (60, 35), (560, 35), (590, 25)],
    first_table_statistic="median",
)
```

- `first_table_xy=None` (default) disables the feature. Otherwise supply 2–12
  distinct nonnegative XY pairs. A strict majority must vote into one connected
  current Lab appearance component; the points can be on unchanged-color table.
  Out-of-view or unusable-color anchors do not vote, and their absence does not
  reduce the required majority. Do not place the points on cloth, the image
  boundary, or an RGB/depth registration edge.
- `first_table_statistic` selects the **signed empty minus current depth**
  estimator on the valid inset of that region, not a statistic of the anchor
  pixels. `"median"` is the default and safer with depth outliers; `"mean"`
  includes the region's valid extremes and can shift with outliers. The offset
  is subtracted from signed depth changes, equivalent to adding it to current
  depth. It is a scalar offset, not a fitted plane, and does not change either
  input image or the RGB comparison.
- Guards include minimum area, valid coverage/pixel count, spatial support,
  and depth agreement around each voting point. The matched empty RGB must
  also agree in encoded Lab chroma, using `background_chroma_tolerance` at the
  votes and the inset's 90th percentile. This rejects a conspicuously different
  hue over the anchors, but allows brightness changes. It can abstain after
  a strong color cast or white-balance change. `first_table_min_anchor_span_px=16`
  requires minimum Euclidean separation between the most distant voters.
  `first_table_max_offset_m=0.008` caps the absolute offset (and median).
  `first_table_max_deviation_m=0.003` caps robust depth dispersion/tile
  mismatch, and mean/median disagreement when using mean. The voters' local
  depth neighborhoods must agree within the smaller of that limit and
  `first_min_depth_delta_m`. Sufficiently strong conflicts or insufficient
  samples leave ordinary color-first behavior in place.
- Only measured, color-compatible pixels close to the robust table depth are
  forced to background. Invalid, too-high, or separated depth is not made
  valid or forced to background. An accepted calibration first tries an
  observed anchor-derived ring local to the object. When calibrated depth is
  spatially consistent to within `first_min_depth_delta_m`, a bounded
  additional search can use the same table within 3× the normal radius, capped
  at 128 px. Without sufficient samples it uses the ordinary ring, not a
  global table value. The anchor-derived reference excludes measurable
  elevation; the robust calibrated median and tile spread bound its baseline
  and noise, so trimming positive values cannot manufacture zero-noise or a
  lower reference. The region's mean, median, and supporting-block gates
  still all apply. The selected offset estimator does not replace these gates.
- This assumes a still-registered view and caller-verified table anchor zones.
  If cloth covers the anchors and is indistinguishable from the table under
  color and paired-depth evidence, the method can misidentify it. In
  particular, coherent shallow cloth with similar chroma covering all
  anchors can mimic a brightness change and a small scalar depth offset. Do not use
  the label as verified material, table geometry, or safe grasp evidence.
  Table-like noise can also leave some pixels unresolved; larger real shifts
  or spatial drift can cause the optional calibration to abstain. No real
  camera accuracy is established by the synthetic examples.

The `table_reference` object at `result.stages["fusion"]` reports the
status, points and votes, chosen signed offset in meters, mean/median, depth
spread and table pixel counts. The per-region `table_reference_source` is
`anchor_local_ring`, `anchor_expanded_ring`, or `ordinary_ring`; use
`table_reference_radius_px` to see the actual search distance. With diagnostics
on, compare `first_table_proposal_mask`, `first_table_reference_domain_mask`,
`first_table_reference_mask`, and `first_table_forced_background_mask`.
`first_candidate_mask` still shows pre-fusion color candidates, including
forced table pixels; forced pixels are removed before region labeling.
`depth_difference_m` and the main `depth_foreground_mask` use the corrected
residual when calibration is accepted, including on an ordinary-ring fallback; `first_table_raw_depth_difference_m` and
`first_table_raw_depth_foreground_mask` preserve the raw comparison. Pixel
decision 8 is the anchor-supported background heuristic. See
[`examples/example_rgbd_color_first_table.py`](examples/example_rgbd_color_first_table.py).

## Diagnostics and GUI

Choose `color_first` in the GUI's mode selector; all constructor parameters are
editable. With `diagnostics=True`, inspect:

- `first_candidate_mask`, `first_color_group_map`, `first_region_label_map`;
- `first_interior_mask`, `first_surround_mask` (unions across evaluated regions);
- `first_mean_delta_m`, `first_median_delta_m`, `first_required_delta_m`;
- `first_failed_*_mask` for independent, overlapping failed gates;
- `first_insufficient_mask` and `first_upper_excluded_mask`.

Per-region `result.stages['fusion']['regions']` (also
`diagnostics['first_statistics']`) contains area, coverage, block counts,
interior/reference distribution summaries, relative statistics, required margin,
accepted flag and failed gate names. NaN maps / null statistics mean a gate was
not evaluable, not that it passed. Rings can overlap between regions. The normal
debug image (`result.save_debug("debug.png", color=color)`) gives final decision
reasons. Individual maps are available through `result.diagnostics` and the GUI.

With a surround override, the per-region diagnostics also report
`surround_min_blocks`, `surround_valid_pixels`,
`surround_min_reference_pixels` when the pixel fallback is considered, and
`surround_reference_source` (`block_means`, `ring_pixels` or null if not
evaluated) with `surround_noise_m`. A failed pixel sample floor appears in
`first_failed_surround_reference_mask`. At `0` the surrounding count failure
mask can be clear even when the reference coverage or pixel sample floor fails.

## Limits

- A region-local coherent depth bias plus color change can exactly mimic a thin
  object. A single current/reference pair cannot distinguish these cases.
- If most object's samples quantize to zero, median separation can fail despite a
  small positive mean. This is intentional conservatism, not absence of an object.
- Correlated noise raises the reference spread and can reject real thin objects.
- Narrow regions, heavy invalid depth, a fully cropped-in object with no table,
  or illumination changes covering the whole reference ring return unknown.
- Identical-color touching shadows may merge with a genuine object and pass the
  whole-region test. Color gradients can split an object into smaller regions.
- An unchanged/background-like object has no color candidate; no depth-only
  fallback is performed. Different colors touching can be separated only when
  the configured color metric distinguishes them.
- Automated tests use synthetic images. Real sensor reliability, parameter
  stability and timing require representative registered real captures.
