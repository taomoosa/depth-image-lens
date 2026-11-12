# Locally depth-supported color segmentation

`RGBDBackgroundSegmenter(fusion_mode='color_supported')` is an experimental,
opt-in alternative to `color_seeded`. The default remains `union`; existing
modes keep their selection rules. It does **not** require a chromatic starting
point: strong local depth can introduce a gray-object seed. Every selected
region must originate from measured, locally supported depth. A color-only
shadow can no longer create a seed by itself.

This is a color-led hypothesis, not verified material segmentation. Similar
colored touching shadows, tiny holes and coherent depth errors remain
counterexamples. All inferred pixels stay uncertain. Even the `confident_*`
rule categories are not probabilities or authorization for robot motion.

## Use

```python
from depth_image_lens import RGBDBackgroundSegmenter, OuterContourDetector

segmenter = RGBDBackgroundSegmenter(
    fusion_mode='color_supported',
    depth_threshold_m=.005,
    supported_anchor_ratio=2.,
    supported_growth_radius_px=4,
    supported_color_tolerance=18.,
)
result = segmenter.detect(
    color, depth, background_color=empty_color, background_depth=empty_depth,
    color_order='RGB', registered_to_depth=True, depth_scale_m=.001,
    diagnostics=True,
)
outline = OuterContourDetector().detect(
    result.object_mask, observed_mask=result.outline_observed_mask,
)
# Preserve uncertainty for downstream consumers; inferred shape is not depth.
result.save_debug('color-supported.png', color)
```

The [registered RGB-D input contract](RGBD_BACKGROUND.md#input-contract) applies.
Use .001 for millimeter-valued depth, 1 for meters. Both frames and modalities
must already be aligned. Run `python examples/example_rgbd_color_supported.py`
for generated examples, or add `--debug /tmp/color-supported.png`. Debug saving
refuses to overwrite an existing file. No new dependency is required.

## Selection rules

1. Retain the original color and depth evidence. `depth_foreground_mask` still
   means signed empty-minus-current Z exceeds `depth_threshold_m`. Camera Z is
   not table-normal height. A raw depth pixel outside the threshold margin
   must also have locally agreeing foreground-depth neighbors. The existing
   `depth_support_radius_px`, `depth_support_fraction` and
   `depth_support_tolerance_m` define this support, with outside/invalid positions
   contributing zero. Isolated spikes cannot seed at the defaults.
2. Make seeds from that support when **either** depth exceeds
   `supported_anchor_ratio * depth_threshold_m` **or** full-Lab background
   subtraction supplies reliable color foreground. Luminance can confirm weak
   depth; chroma is not mandatory. Remove seed components smaller than
   `supported_min_seed_area`. Component connectivity is eight-neighbor.
3. Grow seeds at most `supported_growth_radius_px` eight-neighbor steps. A
   candidate must match its original seed's full encoded-Lab color within
   `supported_color_tolerance`, and also match its previous path pixel within
   that tolerance. Each seed retains its own prototype; there is no global
   average color. Growth does not reset its prototype to progressively drift
   into unrelated colors. The nearest arriving step is fixed; if multiple
   sources arrive together the closest color wins, with deterministic ties.
   This is bounded multi-source growth, not a global optimal assignment.
4. Growth also needs either raw foreground depth or a locally relaxed
   background score greater than `supported_relaxation_ratio ** 2`. This changes
   eligibility only near seeds, not the global color-foreground mask. Reliable
   unchanged-reference color at near-table depth blocks growth. Near-table
   means absolute depth difference at most `depth_margin_m`. Invalid depth,
   invalid current color and farther depth cannot be selected or crossed.
5. Finally veto any selected connected region whose measured, locally supported
   foreground-depth fraction is below `supported_min_depth_fraction`. This
   whole-region check only **rejects** already bounded candidates. It never
   adds all pixels of a color-difference component. Thus a supported object
   does not legitimize an arbitrarily large connected shadow.
6. Apply `min_object_area` to final regions. Filtered pixels remain uncertain;
   accepted seed/inference masks and counts reflect the filter.

Missing depth is always unknown, irrespective of `missing_depth_policy`.
It is not silently treated as desk depth. Near-table color changes that have no
supported seed are excluded from `object_mask`, but remain unresolved: a thin
real object can also produce that evidence. A rejected/invalid empty-color
reference disables subtraction and color-only expansion. Current-color-valid
strong depth can still seed; its selection remains uncertain.

## Settings and tuning

All numeric settings are finite, validated, and included in result parameters.
The six `supported_*` settings affect only this mode.

| Parameter | Default / range | Effect |
|---|---|---|
| `supported_anchor_ratio` | 2 / [1,4] | Strong depth multiplier. Lowering helps shallow gray objects and admits more coherent noise. Reliable color can independently confirm weaker supported depth above T plus the margin. |
| `supported_color_tolerance` | 18 / (0,441] | Full encoded-Lab distance from a seed and previous path pixel. Larger tolerates shading but admits more shadow/color leakage. These are not standardized Delta E units. |
| `supported_growth_radius_px` | 4 / integer 0..20 | Maximum path steps from a supported seed. Zero returns validated seeds only. Larger repairs wider weak-depth rims and can expand into similar-colored shadows/holes. |
| `supported_relaxation_ratio` | .2 / (0,1] | Local background ellipse-radius threshold for growth with no foreground depth. Lower helps weak gray contrast but reduces background rejection. It cannot create a seed. |
| `supported_min_seed_area` | 3 / positive integer | Minimum eight-connected seed component size. Larger rejects small supports and loses small, thin or cropped objects. |
| `supported_min_depth_fraction` | .1 / (0,1] | Minimum locally supported depth fraction within each selected region; a veto only. Raising can reject inferred rims together with the whole real object. |

The raw depth threshold/margin, three depth-support parameters, color reference
checks, full-Lab tolerances, validity masks, ROI and final area filter also matter.
`strong_depth_ratio`, `color_growth_max_steps`, `color_relaxation_ratio`, and
`anchored_*` do not control this mode.

Start with a depth-only comparison and measured empty-frame noise. Inspect seed
pixels before changing growth: no seeds means radius alone cannot help. For a
reliable seed with an incomplete rim, increase radius modestly and compare both
real object recall and nearby shadow pixels. Tune source-color tolerance against
both object shading and touching shadows. Keep uncertainty; never clear it to
force a downstream grasp. Pixel parameters depend on resolution and crop.

## Result evidence and debugging

Original RGB/depth evidence remains unchanged. `observed_mask` is paired depth
availability within ROI; it does not imply trustworthy current RGB.
`uncertain_mask` and `unknown_mask` are equal in this mode. They include all
inferred pixels, depth-only seeds, missing modalities, threshold uncertainty and
unselected color/depth candidates. Neither is a complement of `object_mask`.
`outline_observed_mask` retains selected uncertain geometry while excluding
unresolved nonselected areas.

- `confident_foreground_mask`: retained seeds with reliable color and local
  foreground depth. A coherent erroneous depth/color pair can satisfy it.
- `confident_background_mask`: reliable unchanged-color, near-table growth
  guards, not a proof that there is no sub-resolution object.
- `inferred_foreground_mask`: selected non-seed pixels, always uncertain.

Unlike depth_anchored, these masks do not partition the ROI. Pixels without a
candidate can be neither confident nor unknown. No-change is not verified empty.

With `diagnostics=True`, inspect `supported_seed_mask`, `color_seed_mask`,
`depth_seed_mask` (final retained seeds), `raw_supported_seed_mask` (before seed
area filtering), `growth_candidate_mask`, `relaxed_color_mask`, `growth_step`
(-1 not reached, 0 seed, positive bounded path length), `support_rejected_mask`,
`unsupported_color_mask`, `background_guard_mask`, and
`region_depth_support_fraction`. The last fraction is computed on pre-final-area
selected connected regions, zero outside them; it is not a probability.
`growth_step` also retains routes rejected by later support/area filtering.

The debug decision panel identifies validated color/depth seeds, unsupported
color/depth, unresolved evidence, area filtering and inferred color growth.
No accepted-color-only seed exists. Metrics include retained seed/grown counts
and rejection counts. A zero rejection count is not accuracy certification.

## Comparison and limits

[depth_anchored](RGBD_ANCHORED.md) starts with spatial depth geometry and refines
continuous scores with current RGB edges. This mode keeps local source colors
and finite paths, with depth as seed evidence and a region veto. Neither replaces
the other universally, and both remain opt-in.

Known failures are explicit tests:
- Similar-color touching shadows within the growth radius can be inferred.
- A same-color small real hole can be inferred as foreground. Missing holes are
  left unknown; a background-colored observed hole is blocked by its color.
- Coherent false foreground depth can seed an entire shadow, even into the
  confidence category. Spatial support does not remove persistent bias.
- Thin/unseeded/shallow objects, strong color shading, weak reference contrast,
  wide depth-invalid rims, and misregistration can be lost or distorted.
- Many small noisy seeds can survive. This is not a learned arm/shadow detector.

Synthetic fixtures verify contracts and expose these failures, not real-camera
accuracy. Runtime and temporary memory scale with image area, support-window
area and growth radius. Benchmark at the actual resolution; no real-time claim.
