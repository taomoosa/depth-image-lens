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
| `first_min_blocks` | 4 | Minimum usable interior AND surrounding blocks. Smaller values accept thinner/cropped regions with weaker evidence. |
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
