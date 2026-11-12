# Depth-anchored RGB-D refinement

`fusion_mode='depth_anchored'` is an experimental, opt-in mode of
`RGBDBackgroundSegmenter`. Depth establishes foreground and background evidence;
current RGB, including luminance, helps position uncertain boundaries. It can
find gray objects without chromatic seeds and protect supported narrow parts.
The default remains `union`. Existing four modes retain their selection rules.

**This is a heuristic, not a calibrated probability, material classifier,
registration correction, or safe-grasp guarantee.** Even the returned
`confident_foreground_mask` can be wrong when a shadow and coherent depth bias
imitate a raised object. A corrupted depth hole can be filled. Same-color and
weak-color boundaries can shrink. No physical topology preservation or real
sensor accuracy is promised.

## Minimal use

```python
from depth_image_lens import RGBDBackgroundSegmenter, OuterContourDetector

seg = RGBDBackgroundSegmenter(fusion_mode='depth_anchored').detect(
    current_color, current_depth,
    background_color=empty_color, background_depth=empty_depth,
    color_order='RGB', registered_to_depth=True, depth_scale_m=.001,
    diagnostics=True,
)
outline = OuterContourDetector().detect(
    seg.object_mask, observed_mask=seg.outline_observed_mask,
)
seg.save_debug('anchored.png', current_color)
# Keep seg.uncertain_mask for downstream segmentation_uncertain_mask checks.
# Do not replace measured depth with seg.object_mask or a refinement score.
```

Run `python examples/example_rgbd_anchored.py`, or add
`--debug /tmp/anchored.png`, from an installed checkout. Debug export refuses
to overwrite an existing file; choose a new path when repeating it. The example includes a
gray object, an observed hole, a one-pixel appendage, a dark shadow and missing
depth. It uses only the existing NumPy and OpenCV dependencies.

Inputs use the [existing registered RGB-D contract](RGBD_BACKGROUND.md).
Both depth images must be positive, finite, on one registered grid, with a
shared explicit scale and truthful sensor-validity masks. Empty-current depth
is optical-axis camera Z, not table-normal height. No temporal frames are
consumed and no motion or sensor-noise calibration is performed.

**Missing depth is always unknown in this mode**, even if the legacy
`missing_depth_policy='color'` default is retained. `observed_mask` equals paired
depth validity intersected with the ROI. No missing sample is filled or crossed.
Farther-depth pixels are excluded and cannot be crossed during domain growth.

## Evidence and refinement

1. Preserve raw depth difference and foreground masks. For optimization only,
   divide depth difference by `depth_threshold_m` and clip to [-4, 4]. Compute
   a 3x3 median using available neighbors whose current encoded-Lab distance
   is at most twice `anchored_color_sigma`. Where current RGB is unavailable,
   use neutral color compatibility. This robust array is not measured depth.
2. Establish foreground anchors in eroded robust-depth interiors with strong
   raw depth. A second interior path accepts above-threshold depth whose raw
   3x3 range is within `depth_support_tolerance_m`. Where current color agrees
   with the reference, require a larger interior margin. Spatial consistency
   is not temporal stability and cannot reject persistent bias.
3. Protect thin parts separately. Opposite, supported raw-depth neighbors with
   lower-depth transverse neighbors establish narrow runs. Weak runs without
   a visible transverse RGB edge require ten-times-tighter depth agreement.
   A visible narrow RGB run can also use a five-sample directional median with
   at least three valid, color-compatible samples; reference-color agreement
   disables this extra path. Above-threshold center depth is still required.
   Endpoints need an adjacent supported run. Small raw-depth components of
   3–25 pixels with at least 80% strong pixels have a separate anchor path.
   Singletons, very weak small components, and noisy same-color thin parts
   remain unresolved or can be lost. Pixel-scale rules are resolution-dependent.
4. Keep robust-depth components containing a positive anchor, then expand
   the refinement domain by at most `anchored_refine_radius_px` 8-neighbor
   steps through valid, non-farther depth. This limits inference, not the size
   of an already depth-supported component. Unseeded regions cannot originate
   from color alone. This is not global flood fill or morphological closing.
5. Fix background anchors at supported near-table depth and at farther depth.
   A separate raw-background guard protects small holes when raw near-table
   depth agrees with reliable reference color, even if a median would erase an
   isolated table sample. Near-table
   means magnitude below half the depth threshold. Without robust agreement,
   this veto remains unknown, not a supported/confident background anchor. A 3x3 interior
   with both raw and robust near-table depth also anchors background. Forced domain
   exclusions are recorded separately and are not confident background evidence.
6. Relax a continuous score between fixed anchors with four-neighbor weights
   `anchored_pairwise_weight * exp(-Lab_distance_squared / (2*sigma*sigma))`.
   All three encoded-Lab channels contribute, including L. The unary target is
   `clip((robust_depth_ratio - 1)/3 + .5, 0, 1)` with fixed weight .2. Each Jacobi
   iteration averages that target and neighbor scores. Select scores above .5
   inside the permitted domain. This is bounded iterative diffusion, not exact
   graph-cut optimization; convergence is not guaranteed within the budget.
7. Apply the existing optional `min_object_area` filter. Every selected
   non-anchor pixel remains uncertain. Raw evidence and pre-filter anchors
   remain inspectable rather than being overwritten by final selection.

Current-image edge availability is independent of the empty-color reference:
rejected reference checks disable subtraction evidence and related confidence,
but valid current RGB edges remain usable. Black/white clipped pixels likewise
retain current-image edge information; they do not gain reliable subtraction
confidence. `color_valid_mask` controls current edges, while
`background_color_valid_mask` only affects the reference comparison.

## Settings

All settings are validated and recorded in every mode. These five only affect
`depth_anchored`:

| Setting | Default and allowed values | Meaning and tradeoff |
|---|---|---|
| `anchored_color_sigma` | 12; finite (0,441] | Current encoded-Lab affinity scale and local compatibility. Larger values cross more color edges and can erase detail. Not a physical color-noise estimate. |
| `anchored_pairwise_weight` | 2; finite [0,100] | Relative neighbor influence against unary weight .2. Larger values smooth more, possibly shrinking/expanding boundaries. Zero leaves unary selection plus anchors/domain constraints. |
| `anchored_refine_radius_px` | 4; integer 0..20 | Maximum domain expansion from retained depth-supported components. Zero still allows refinement within those components. Larger values permit more uncertain geometry. |
| `anchored_iterations` | 24; integer 1..100 | Maximum Jacobi updates, with early exit below maximum score change 1e-5. More iterations may change selection; no accuracy guarantee. |
| `anchored_anchor_ratio` | 2; finite [1,4] | Strong raw depth is above this multiple of `depth_threshold_m`. Does not disable the spatially consistent shallow, or supported narrow-run, alternative paths. |

`depth_threshold_m`, `depth_support_tolerance_m`, reference-check/tolerance
settings, validity, ROI and area filtering also matter. The support tolerance
controls raw spatial consistency; increasing it admits more noise. The old
`depth_support_radius_px`, `depth_support_fraction`, `strong_depth_ratio`, and
`color_growth_*` settings do not control this mode. No automatic depth-noise,
registration-error, physical-size or camera-distance normalization occurs.

Before tuning, compare raw depth-only selection, RGB subtraction and the new
anchor/inference maps. Estimate noise and registration uncertainty using actual
captures. Lower thresholds can recover shallow objects but also admit shadows
with depth errors. Increasing color smoothness is not a general cure for noise.

## Results, unknown regions and diagnostics

The existing masks retain their meanings. Four additional result masks are
available even with `diagnostics=False` (they are `None` in older modes):

- `confident_foreground_mask`: selected interior body anchors with reliable
  foreground reference-color evidence, outside the raw depth threshold margin
  and away from the selected boundary.
- `confident_background_mask`: supported near-table background with reliable
  reference-color agreement.
- `unknown_mask`: ROI minus those two disjoint categories. It equals
  `uncertain_mask` in this mode and includes missing depth, unresolved changes,
  same-color foreground, thin-only/small-only anchors and all inferred pixels.
- `inferred_foreground_mask`: final selected pixels that were not foreground
  anchors. This is selected geometry, not observed foreground depth.

The three confidence/unknown masks partition the ROI. **Confidence is only a
rule category**, not independently verified correctness. Correlated errors can
produce confident false positives; same-color objects below depth resolution
can look confidently background. The output does not authorize robot motion.

`outline_observed_mask` keeps selected uncertain geometry for contour extraction,
but excludes nonselected unknown pixels, suppression and deferral. Missing cuts
and resulting binary holes are not verified physical openings. Do not clear
uncertainty to force a downstream pick result.

Diagnostics retain `depth_difference_m`, original `depth_foreground_mask` on the
result, color subtraction maps, and these new entries:

- `robust_depth_ratio`: clipped dimensionless local estimate, not raw metres.
  Undefined outside `depth_valid_mask` (stored as zero there).
- `current_color_edge_valid_mask`: validity of current RGB edges.
- `foreground_anchor_mask`, `thin_depth_anchor_mask`,
  `small_component_anchor_mask`, `background_anchor_mask`: pre-area-filter
  optimization evidence; anchors can overlap within the foreground categories.
- `raw_background_guard_mask`: raw table-depth counterevidence with reliable
  reference agreement but no robust agreement. This veto is explicitly unknown,
  not a supported background anchor. Rejected/clipped references disable it.
- `refinement_domain_mask`, `refinement_excluded_mask`: permitted and forced-out
  paired-depth pixels. Exclusion is not physical background confidence.
- `refinement_score`: final numeric optimization score in [0,1], not probability.
- Final confidence, unknown and inference masks; `fusion_decision` and legend.

Metrics include counts, actual iterations, `refinement_converged` and
`refinement_max_update`. A budget-exhaustion warning is emitted when the maximum
score update has not dropped below 1e-5. An empty mutable domain terminates in
one iteration. Anchor counts are pre-filter; final inferred/confident counts
reflect the area filter. Debug PNGs show a mode-specific decision panel alongside
raw depth and color scores. Numeric refinement arrays remain available separately.

Decision codes: 0 = no selected change/unavailable, 1 = body or small supported
depth anchor, 2 = thin anchor, 3 = background anchor, 4 = unselected raw depth
candidate, 5 = other unresolved boundary/color candidate, 6 = area-filtered,
7 = refined foreground inference. Later decisions take precedence. Code 3 is an
optimization constraint, not by itself a confidence label; consult the masks.

## Known limits and validation scope

Tests cover gray/luminance objects, shadows, thin lines and endpoints, small
components, true holes, invalid/missing/farther depth, ROI cuts, clipped color,
rejected references, retained raw evidence, optional diagnostics, area filtering,
units/strides, legacy-mode compatibility and explicit correlated-error failures.
The repository's evaluation policy keeps larger comparison runners and generated
reports outside the public source tree.

Work and temporary allocations scale with the full image area, not just the
refinement band. Directional sample stacks, shifts and iterative score buffers
can use substantial memory at large resolutions. No real-time performance is
claimed; measure latency and peak memory in the intended process and resolution.

Necessary counterexamples include:

- Color-only shadows do not create foreground anchors, but a persistent depth
  error can satisfy every foreground rule and even the confidence category.
- An isolated valid-but-wrong table-depth sample inside a same-color solid
  object is indistinguishable from a one-pixel hole. The raw-background guard
  can produce a false hole, retained as unknown rather than verified topology.
- If the depth inside a real hole is corrupted toward the foreground, the hole
  can be selected. No topology or material truth is inferred from enclosure.
- Below-threshold shallow objects may disappear; random noise can still create
  a few false supports. Weak same-color boundaries and tiny components can be
  lost even when a depth-only threshold selects them.
- One-pixel detail with sharp RGB can survive, but blurred/missing depth and no
  usable color edge can still destroy it. Protecting detail is not universal.
- RGB-D shifts can improve or worsen boundaries. The method does not estimate
  or correct registration. Shadows themselves can provide strong color edges.
- Iterative smoothing may shrink weak boundaries, join nearby candidates or
  leave disconnected islands. It does not enforce connectedness or physical
  holes. Increasing the domain, iterations or pairwise weight changes tradeoffs.

Synthetic comparisons test these mechanisms, not deployment precision/recall.
Validate on independently annotated real RGB-D pairs, at the actual resolution,
materials, pose, lighting and sensor settings before selecting this mode.
