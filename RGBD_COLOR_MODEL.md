# Fixed region color models for RGB-D boundary refinement

`RGBDBackgroundSegmenter(fusion_mode="color_model")` is an opt-in alternative to
`color_local`. It re-estimates a color candidate region from depth-supported
interior colors and the registered empty-scene color, then applies bounded
spatial connectivity. It is not a semantic segmentation model or proof of grasp.
Existing modes and defaults remain unchanged.

```python
segmenter = RGBDBackgroundSegmenter(
    fusion_mode="color_model",
    local_seed_min_area=256,       # preserve the chosen noise-area filter
    model_color_tolerance=18,
    model_background_margin=3,
    model_boundary_radius_px=6,
)
result = segmenter.detect(
    color, depth, background_color=background_color,
    background_depth=background_depth, color_order="RGB",
    registered_to_depth=True, depth_scale_m=0.001,  # integer millimetres
    diagnostics=True,
)
mask = result.object_mask
```

Use `depth_scale_m=1` for metre depth arrays. The color/depth/background grids
must already be registered. Color order must be explicit.

## What changes

1. Reuse `color_local` signed, color-weighted local depth statistics and its
   seed-area filter. Do not lower foreground evidence thresholds just to repair
   an edge.
2. Within each connected seed region, use only valid center depth beyond
   `depth_threshold_m + depth_margin_m` to train a fixed palette. Dense color
   modes are located with coarse bins, and represented by robust medians of
   bounded training clusters. Multiple supported colors are kept separately.
3. A candidate pixel must be close to a palette color **and** closer to that
   color than to its own same-position background color, by a margin:

   `distance_to_model <= model_color_tolerance`

   `distance_to_model + model_background_margin < distance_to_background`

   Both distances are full OpenCV encoded-Lab Euclidean distances, not standard
   Delta E or probabilities. The original relaxed background-change candidate
   condition also remains required.
4. Recompute connectivity in this fixed color domain, starting from the seed
   region, with a maximum geodesic boundary distance. Models are never updated
   using inferred pixels, avoiding self-reinforcing color drift.
5. Unlike `color_local`, locally table-like pixels may be rescued **inside this
   bounded band**. These are inferred/unknown, not observed depth support.
   Invalid color/depth, farther depth, ROI exclusions, and the optional
   `local_max_reference_separation_m` upper band remain hard exclusions.

This specifically trades some shadow protection for smoother, color-supported
edges. A same-color touching shadow that matches the model and differs from the
background can also pass within the boundary band. The method cannot distinguish
it from a real edge with the same RGB-D observations.

## Main parameters

| Parameter | Default | Tuning |
|---|---:|---|
| `model_boundary_radius_px` | 6 | Maximum 8-neighbor distance from accepted source pixels. Try 3, then 6; larger recovers wider depth-dropout edges but leaks farther into matching shadows. 0 disables boundary expansion. |
| `model_color_tolerance` | 18 | Maximum distance to a learned palette color. Increase for within-object color variation; this may admit shadows/other objects. |
| `model_background_margin` | 3 | Required advantage over the same-pixel background. Larger rejects ambiguous background-like pixels but loses weak-contrast real edges. |
| `model_min_training_pixels` | 8 | Required strong-depth training count per seed region. Increasing rejects unstable small models and small objects alike. |
| `model_max_colors` | 6 | Maximum dense color modes per region (1..16). A single average is not used across different colors. |
| `model_min_color_fraction` | .03 | Minimum training proportion per mode, also at least three pixels. Rare colors may be omitted. |
| `model_max_regions` | 32 | Process largest seed regions first; skipped regions stay unresolved, not background. |

`local_seed_min_area`, local support/mean/density settings and
`depth_threshold_m` still control the interior evidence. The local color
relaxation setting controls initial candidates. `local_growth_radius_px` and
`local_color_tolerance` belong to `color_local` growth and do **not** control
`color_model` output; use the `model_*` values above. Local table-like thresholds
remain useful diagnostic labels but no longer veto model inference in the band.
These are explicit parameters, not automatic empty-table noise calibration.

## Diagnostics

- `result.stages["fusion"]["models"]`: training counts, palettes, selection status.
- `model_training_mask`: selected strong-depth pixels used by a model.
- `model_source_mask`: accepted starting pixels from local seed regions.
- `model_rescued_background_mask`: selected pixels locally classified table-like.
- `model_rejected_core_mask`: seed-core pixels rejected by the color tests.
- `model_unavailable_mask`: missing training/palette or model-budget omissions.
- `model_label_map`, `model_color_distance`, `model_background_distance`.
- `growth_step`: 0 at sources, positive at bounded inference, -1 unselected.
- Local maps remain available for diagnosing mean, density and seed proposals.

All selected non-training pixels are in `inferred_foreground_mask` and remain
unknown. Existing color-reference and depth-threshold-margin uncertainty is
preserved. Area filtering updates accepted provenance masks. `model_budget_exceeded`
is exposed in metrics. Unavailable/omitted models do not justify a no-object claim.

## Limits and safe use

- A depth-biased table patch large enough to pass the seed-area filter can train
  a false object model. A 256-pixel threshold is not a safeguard against broad
  bias or connected errors.
- Same-color touching shadows can be accepted inside the boundary band. Do not
  increase the radius indefinitely to remove every hole.
- Colors not present in supported training pixels may remain missing, including
  an entirely depth-missing second object color or weakly observed thin objects.
- An object visually indistinguishable from its background may be rejected.
- Missing depth remains missing. No sensor depth is synthesized or marked valid.
- An invalid-depth cut, real opening, color-incompatible narrow bridge or ROI cut
  can break propagation. True physical topology is not inferred reliably.
- Local seed regions may themselves extend past a true physical boundary; the
  radius is measured from source pixels, not ground-truth object edges.
- This mode does not infer grasp force, arm identity, registration, or noise bias.
  Every recovered edge is a hypothesis for downstream consumers to handle.

## Reproducible evaluation

Run `PYTHONPATH=src python examples/example_rgbd_color_model.py`. It compares
`color_local` and `color_model` with identical shared settings and fixed seeds.
It includes edge depth dropout, same-color shadows, color variation, shallow
noisy objects and coherent false seeds. The accompanying unit tests cover
multiple colors, disconnected matching regions, fixed-model drift protection,
ROI, missing observations, upper/farther depth, small models, model budget and
provenance. Real-image accuracy has not been validated.

### Measured synthetic results

The example uses a 90x140 image, a gray rectangle, a 5-pixel edge band with only
0.4 mm depth difference, a 6 mm interior, and independent 1 mm Gaussian noise.
Shared settings are raw threshold 2 mm, margin 0.5 mm, seed area 256 and local
background mean 0.25 mm. The thin case instead uses 3 mm and 4 mm noise; the false
seed case adds 6 mm depth bias to the shadow. Every result below averages ten
fixed seeds. These are specific operating points, not optimally tuned benchmarks.

| Case | color_local object / edge / shadow | color_model object / edge / shadow |
|---|---|---|
| Edge dropout, different-color shadow | .893 / .660 / 0 | .997 / .991 / 0 |
| Same-color touching shadow | .891 / .653 / 0 | .997 / .991 / .058 |
| Mottled object | .971 / .909 / 0 | .997 / .991 / 0 |
| Thin noisy object | .896 / .892 / .003 | .896 / .893 / 0 |
| Coherent false depth seed | .909 / .713 / 1 | .997 / .991 / 1 |

For the same-color shadow, boundary radii 2 / 6 / 10 gave edge recall
.653 / .991 / .991 and shadow acceptance 0 / .058 / .156. Increasing radius beyond
the dropout width did not improve recall but enlarged shadow leakage. The total
binary grid perimeter was 250.4 / 283.8 / 313.2 pixels, respectively. Perimeter
alone is not a smoothness score because coverage and false extensions differ.
There is **no unconditional smoothness guarantee**: a distance cutoff, protected
depth discontinuities and true color noise can still leave a jagged edge. This
mode improves color-consistent edge completeness, not arbitrary geometric shape.

On the test machine (Intel Core i7-14700), 640x480 diagnostics-enabled calls after
one warm-up had median times over three runs of 0.71 s (`color_local`) and 0.76 s
(`color_model`), excluding capture/rendering. Cost grows with local support
radius and model count; remeasure on the target workload.

## Diagnosing rejected pixels

Run `PYTHONPATH=src python examples/example_rgbd_color_model_diagnostics.py`.
The `explain_pixel(result, y, x)` helper reports each evaluable model separately.
Enable `diagnostics=True`; these extra arrays are not computed otherwise and do
not change segmentation. The fusion image retains its existing colors. Its new
text panel counts overlapping failed tests over unselected pixels for **any**
evaluable model, not a unique cause or a vote. One model can reject a pixel that
another accepts. Use the per-model arrays for attribution.

Red decision 4 means raw depth is foreground but the pixel was not selected;
it overwrites decision 3, so it does not identify the actual rejecting test.
Neither red nor cyan means an enclosed hole. Check the final `object_mask` and
connectivity before applying geometric repair.

New diagnostics have shape `(number_of_evaluable_models, H, W)` and matching
`model_diagnostic_ids`. IDs reference `stages['fusion']['models']`:

- `model_all_color_distances`: nearest palette distance for that model at every
  usable color pixel, including rejected pixels. NaN means unusable color,
  not zero distance. Existing `model_color_distance` remains selected-only.
- `model_all_source_distances_px`: exact obstacle-free 8-neighbor (Chebyshev)
  distance to that model's color-qualified source. NaN means no source. This is
  a lower bound on path length, **not** the color-constrained geodesic distance.
- `model_all_reachable_masks`: connected to a source through the complete valid
  color domain, ignoring the step limit. False can also mean color/domain failure.
- `model_all_selected_masks`: that model's selection before global area filtering.
  Always consult final `object_mask` / decision 6 for area-filtered pixels.
- `model_all_rejection_flags`: independent overlapping bits listed in
  `stages['fusion']['model_rejection_flags']`. Bits 1/2/4/8/16/32 indicate ROI,
  color usability, depth validity, relaxed color candidate, farther and upper
  exclusions. Bits 64/128 are model distance and same-position background
  competition failures. Bit 256 means this model has no qualified source.
  Bit 512 means color-qualified but disconnected; bit 1024 means connected but
  beyond the actual bounded growth. These last two require a source and valid
  color/domain, and are mutually exclusive. Color failures are evaluated only
  where color is usable. Multiple input and color failures may coexist.
- `model_no_evaluable_model_mask`: ROI pixels when no palette can be evaluated.
  Empty stacks have shape `(0,H,W)`, not an invented infinite distance. Inspect
  model records for insufficient training or missing palette, existing
  `model_unavailable_mask` for affected seed regions, and
  `model_budget_exceeded` for omitted models. A distant evaluable model does not
  erase an unavailable local model's status.

`model_background_distance` is the existing same-position full-Lab difference;
ignore it on color-unusable pixels. Compare a model's color distance plus
`model_background_margin` against this value. Do not combine the smallest color
distance from one model with reachability from another. No model is inferred
for unsupported training regions. These are algorithmic reasons, not semantic
proof that a pixel is shadow or object.

Diagnostic memory grows with model count (approximately 12 bytes per pixel per
model for the retained new stacks, plus temporary allocations). At 640x480 with
32 evaluable models the stacks alone use about 118 MB. Disable diagnostics in
production when they are unnecessary. The connectivity pass adds diagnostic-only
runtime; it does not change the existing growth radius or mask.

### Spatial rejection tiles

`result.render_debug(color, color_order='RGB')` now appends one spatial tile per
independent test plus a no-evaluable-model tile. Orange marks **unselected**
pixels failing that test, gray is the selected object for context, black is
neither. A pixel can be orange in several tiles: there is no priority coloring
that hides overlapping reasons. All original fusion panels remain present.

By default the tile marks failure for **ANY evaluable model**. It is not a claim
that every model failed that same test; one model can fail color while another
fails connectivity. To inspect a particular model without that aggregation:

```python
model_id = int(result.diagnostics['model_diagnostic_ids'][0])
image = result.render_debug(color, color_order='RGB', model_diagnostic_id=model_id)
```

Check for an empty ID list first. IDs are region identifiers, not array indices.
Unknown IDs, noninteger IDs, and requesting a model without detailed diagnostics
raise `ValueError`. The pre-existing count panel still reports ANY-model counts;
the spatial panel header identifies its own scope. Model records and
`explain_pixel` remain the authoritative per-pixel details. No-source and
non-candidate tests can mark broad regions; they are not evidence of an object.
For no evaluable models, all per-model test tiles are empty and the dedicated
ROI tile is orange. Per-model tiles deliberately omit pixels already accepted by
another model; use the numeric stacks to inspect those disagreements.

Run the example with `--output /tmp/color_model_debug.png`, optionally adding
`--model-id 1`. It writes only when an output path is provided. Spatial tiles use
nearest-neighbor scaling; zoom or inspect numeric masks to check single pixels.
Rendering iterates models per test rather than allocating another full stack;
it needs image-sized scratch buffers and a taller output canvas. A rendered
image is not an additional segmentation result and never alters the masks.

### How background difference affects selection

There is no weighted sum of depth, object color, and background color, and no
independent background weight parameter. Selection must pass separate gates.
With corrected same-position background Lab difference `(dL, da, db)`:

```
dC = sqrt(da**2 + db**2)
S = (dC / background_chroma_tolerance)**2 + (abs(dL) / background_luma_tolerance)**2
relaxed_color_candidate = S > local_color_relaxation_ratio**2

d_background = sqrt(dC**2 + dL**2)
d_model = minimum Euclidean Lab distance to this model's palette
color_ok = (d_model <= model_color_tolerance) AND
           (d_model + model_background_margin < d_background)
```

Candidate, valid input, source connectivity and growth budget must also pass.
Large background difference cannot compensate for an incompatible object color
or disconnected source. The normalized candidate score uses defaults 20 for
chroma tolerance, 60 for luminance tolerance and 0.2 for relaxation. Lowering a
tolerance raises sensitivity to that difference, effectively changing its
inverse-square contribution to this **first** gate. The subsequent full-Lab
background competition does not use those normalization tolerances. Defaults
are 18 for object model tolerance and 3 for the background margin. Reducing the
margin relaxes how much closer the pixel must be to the object model than to
its background, potentially admitting shadows. `background_score_margin` is
uncertainty bookkeeping, not a background competition weight. Distances use
OpenCV encoded 8-bit Lab, not standardized Delta E; extreme normalized scores
are saturated internally for numerical safety.
