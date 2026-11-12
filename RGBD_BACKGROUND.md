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
1 as uncertain; it does not change legacy selection. Evidence mode additionally
uses this margin to gate color confirmation/background agreement and chroma-only selection.

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
| `fusion_mode='evidence'` | Cross-check color and depth, retaining strong supported same-color depth while suppressing unsupported depth with background-color agreement. See the decision rules below. |
| `fusion_mode='color_seeded'` | Preserve strict chroma seeds, then add only nearby relaxed-color pixels with locally supported foreground depth. See [bounded color growth](#bounded-color-growth). |
| `fusion_mode='depth_anchored'` | Depth-supported anchors plus bounded current full-Lab edge refinement, with separate thin/small-component support and explicit unknown/inference masks. See [depth-anchored refinement](RGBD_ANCHORED.md). |
| `fusion_mode='color_supported'` | Locally depth-validated color/depth seeds with bounded source-color growth; no unsupported color-only seeds. See [color-supported segmentation](RGBD_COLOR_SUPPORTED.md). |
| `fusion_mode='color_region'` | Whole 8-connected full-Lab regions tested by valid-depth coverage, clipped mean and support density. No square local support or bounded growth. See [color-region support](RGBD_COLOR_REGION.md). |
| `missing_depth_policy='color'` (default) | In union/evidence/color_seeded modes, retain that mode's color-only candidates where either depth is missing, marked uncertain. |
| `missing_depth_policy='unknown'` | Require valid depth even for color candidates; missing depth is unobserved. In depth_required mode, depth is required regardless of this setting. |

Color-only pixels remain uncertain even when depth is valid. Farther changes are
also uncertain; `conflict_mask` records color foreground at farther-depth pixels.
In legacy modes, valid-depth foreground remains selectable when color is missing or rejected.
Missing either modality and clipped/near-threshold evidence are marked uncertain.
`uncertain_mask` is independent of `object_mask`: uncertainty does not automatically
remove a candidate. Unselected pixels without uncertainty merely lack a detected
change under the declared model; they are not certified background.

No inferred depth filling is applied. The legacy modes do not close, erode or polygon-fill masks.
Color_seeded performs bounded mask growth, described below. The separate
depth_anchored mode performs depth-supported edge refinement; see
[its contracts, parameters, debug masks and limitations](RGBD_ANCHORED.md).
In legacy modes, nonconvex shapes, holes, small thin features and multiple components
are retained as measured. Evidence mode can reject insufficiently supported pixels.
Optional `min_object_area` removes 8-connected components smaller
than that pixel count and marks the discarded pixels uncertain. Default 1 removes
nothing. Diagonally touching objects still form one component.

## Evidence fusion

Opt in with `RGBDBackgroundSegmenter(fusion_mode='evidence')`. The default remains
`union`. This mode changes selection by evidence, not by eroding a fixed boundary
band. It can still lose real features when evidence is insufficient. It does not
align images, estimate temporal stability, certify material edges, or compute
calibrated correctness probabilities.

Color is **usable for confirmation or background agreement** only when the
reference is usable, both color validity masks are True, and the pixel is neither
clipped nor within `background_score_margin` of the Lab ellipse threshold.
Here “reliable” in diagnostic names means those gates passed, not that the color
is physically correct. Unclipped shadows and color registration errors can pass.

Depth foreground still means `dZ > depth_threshold_m`. Strong depth additionally
requires `dZ / depth_threshold_m >= strong_depth_ratio`. Local support counts
neighboring depth-foreground pixels whose **depth difference dZ** agrees with the
center within `depth_support_tolerance_m`. The denominator is the entire
`(2*depth_support_radius_px+1)^2 - 1` neighborhood, excluding the center.
Missing, background, outside-ROI and outside-image positions contribute zero.
This tests the relative change, not smoothness of absolute camera Z; a coherent
reference error can pass. There is no recursive mask propagation or filling.

Rules are applied per pixel; reasons 1–5 are mutually exclusive:

| Code | Evidence | Decision |
| --- | --- | --- |
| 1 | Depth foreground and reliable color foreground | Select; existing threshold/missing uncertainty remains. Luma-only color can confirm depth. Local support is not required in this branch, preserving color-confirmed narrow features. |
| 2 | No color confirmation, but strong AND locally supported depth | Select with uncertainty, including same-color objects. Color background agreement does not veto this branch. |
| 3 | No depth foreground/farther veto, reliable chroma change `dC/Tc > 1 + background_score_margin` | Select a color-only candidate with uncertainty; missing depth follows `missing_depth_policy`. |
| 4 | Remaining depth foreground with reliable background-color agreement: strength OR local support is insufficient | Suppress from object_mask, set suppressed_mask AND uncertainty. This is insufficient evidence for selection, not proof of background. |
| 5 | Remaining depth/color candidate | Defer from object_mask, set deferred_mask AND uncertainty. Includes luma-only changes, unusable color with weak depth, and farther/color conflict. |
| 0 | No candidate or unavailable | No selection; consult availability and uncertainty to distinguish these cases. |
| 6 | Selected pixel subsequently removed by min_object_area | Area-filtered; takes precedence over selection reasons 1–3. |

`conflict_mask` retains its existing narrow meaning: color foreground plus farther
depth. Use `suppressed_mask`, `deferred_mask`, and diagnostic reasons for the new
disagreements. No-color foreground is not itself a conflict: same-color objects exist.

The strong-depth setting applies only to evidence selection. The three depth-support
settings are shared with color_seeded growth and color_supported seed validation; all settings
are validated and recorded in every mode:

See the [segmenter settings](RGBD_PICK_TUNING.md#segmenter-settings) for defaults,
ranges and adjustment directions. Support computation cost grows with the
square neighborhood area.

All numeric settings reject bool, NaN and infinity. Existing Tc/Tl, score margin,
T, missing policy, ROI, reference-check and area-filter settings remain relevant.
For tuning order and failure symptoms see [the tuning guide](RGBD_PICK_TUNING.md#evidence-mode).

```python
seg = RGBDBackgroundSegmenter(fusion_mode='evidence').detect(
    current_color, current_depth,
    background_color=empty_color, background_depth=empty_depth,
    color_order='RGB', registered_to_depth=True, depth_scale_m=.001,
    diagnostics=True,
)
outline = OuterContourDetector().detect(
    seg.object_mask, observed_mask=seg.outline_observed_mask,
)
seg.save_debug('fusion.png', current_color)
reasons = seg.diagnostics['fusion_decision']  # uint8 codes above
support = seg.diagnostics['depth_support_fraction']  # fraction, not probability
```

`observed_mask` still means input availability under the policy.
`outline_observed_mask` is a new computed mask: observed minus suppressed and
deferred pixels. This prevents the rejected/unresolved pixels from being silently
treated as observed background when extracting corners/segments. Use the original
selected `object_mask`, not an eroded mask, and do not exclude **all** uncertain
pixels: selected same-color or color-only shapes would disappear. Unknown cuts
can split contours; their corners/segments are not guaranteed physical edges.
Both new masks are False in legacy modes, so this property equals observed_mask
there. Area filtering keeps its existing behavior and is reported separately.

For pick, pass the resulting outline and `segmentation_uncertain_mask=seg.uncertain_mask`
along with depth validity under the usual pick contract. Selected depth-only and
color-only evidence remains uncertain, so it will not provide trusted pick support;
a wholly same-color shape can produce `unknown` / no pick candidate. Do not clear
uncertainty merely to force a pick. Outline geometry and trusted grasp evidence
are different outputs.

With `diagnostics=True`, reasons, support fraction, strong/support masks, reliable
color foreground/background masks, and accepted-both/depth-only/color-only masks
are recorded. Accepted masks reflect the final area filter. `stages['fusion']`
contains the reason legend; `metrics['decision_counts']` counts each reason.
Suppressed/deferred masks and counts are available without diagnostics. Debug PNGs
include a dedicated color-coded decision panel even if generic mask thumbnails
are truncated. Numeric support fractions are in arrays, not rendered as a heatmap.

Run `PYTHONPATH=src python examples/example_rgbd_fusion.py` for 11 generated cases
and all three modes. `--output-dir /existing/parent/new-directory` saves a comparison
JSON, per-case JSON/NPZ, and PNGs. It explicitly includes failures: true same-color
1-pixel line endpoints can be suppressed; weak same-color objects can disappear;
shadow plus depth bias can satisfy both modalities; colored misalignment artifacts
remain uncertain candidates; strong coherent depth bias can look exactly like a
same-color object. Single pairs cannot resolve these ambiguities. These are rule
comparisons, not actual-pair precision/recall measurements.

## Bounded color growth

`fusion_mode='color_seeded'` is an opt-in alternative for a chromatically distinct
object whose strict color mask has weak edges or small observation holes. It is
not a replacement for same-color depth segmentation. The default stays `union`;
`union`, `depth_required`, and `evidence` selection rules are unchanged.

The mode never smooths the entire mask, fills a hole merely because it is enclosed,
or invents depth. It combines a local threshold relaxation with bounded,
8-connected growth:

1. **Seeds:** usable, unclipped color with `dC/Tc > 1 + background_score_margin`,
   outside the existing color uncertainty margin. Luminance change alone cannot
   seed an object, even with strong depth. Farther depth vetoes seeds. Missing
   depth follows `missing_depth_policy`. Strict color seeds do not need depth
   support, so a thin chromatic line or isolated chromatic pixel can survive.
2. **Growth candidates:** usable, unclipped color with ellipse score
   `(dC/Tc)^2 + (dL/Tl)^2 > color_relaxation_ratio^2`, plus depth foreground and
   the same local dZ support used in evidence mode. This deliberately permits
   colors below the strict threshold or within its uncertainty margin. Growth
   does not require `strong_depth_ratio`, but does require actual valid depth
   above `depth_threshold_m`; missing depth is never repaired.
3. **Bound:** add only candidates reachable from an initial seed in at most
   `color_growth_max_steps` 8-neighbor steps. Every intermediate pixel must pass
   the growth gates. The bound does not reset at newly added pixels. Gaps,
   invalid color/depth and ROI cuts cannot be jumped. At zero steps, return only
   seeds. A diagonal move counts as one step, so this is not Euclidean distance.
4. **Provenance:** all added pixels are uncertain, even with good depth support.
   Apply the existing optional component-area filter afterward. Neither the
   growth nor its uncertainty implies a verified material edge or safe pick.

See the [segmenter settings](RGBD_PICK_TUNING.md#segmenter-settings) for growth
parameter defaults and ranges.

These settings are validated and recorded in all modes but affect only color_seeded.
The existing three `depth_support_*` settings gate growth only, not seeds. A radius
of 1 and fraction .25 require two agreeing depth neighbors: weak one-pixel tips
may still be lost. Work is bounded by image area times the support neighborhood
plus the number of growth steps. No new dependency is added.

The decision legend is mode-specific in `stages['fusion']['decision_reasons']`:
1 = strict seed with depth, 3 = strict seed without depth confirmation,
4 = unselected depth with usable background-color agreement,
5 = other unresolved color/depth/relaxed-color candidate, 6 = area-filtered,
7 = accepted bounded growth, 0 = no candidate/unavailable. Code 2 is unused.
Suppressed and deferred pixels remain uncertain and are excluded from
`outline_observed_mask`, not silently declared observed background.

With diagnostics enabled, inspect `color_seed_mask`, `relaxed_color_mask`,
`growth_candidate_mask`, `growth_step` (-1 not reached, 0 seed, 1..N growth),
`accepted_growth_mask`, depth support, and the decision map. Seed and step maps
retain **pre-area-filter provenance**; accepted masks, decision codes and
`metrics['grown_pixels']` describe the final selection. `metrics['seed_pixels']`
counts initial seeds before area filtering. The dedicated debug panel labels the
selected mode's decisions. `observed_mask` still records policy-dependent input
availability, not the physical certainty of a grown pixel.

```python
seg = RGBDBackgroundSegmenter(
    fusion_mode='color_seeded', color_relaxation_ratio=.5, color_growth_max_steps=2,
).detect(current_color, current_depth,
         background_color=empty_color, background_depth=empty_depth,
         color_order='RGB', registered_to_depth=True, depth_scale_m=.001,
         diagnostics=True)
outline = OuterContourDetector().detect(
    seg.object_mask, observed_mask=seg.outline_observed_mask)
# Keep seg.uncertain_mask for downstream segmentation_uncertain_mask checks.
```

Run `PYTHONPATH=src python examples/example_rgbd_seeded.py` for one minimal API
example.

**Limitations:** No strict chroma seed means no object, even if same-color depth
is strong. Rejected/missing/clipped color cannot trigger a depth-only fallback.
Large color holes and distant weak endpoints remain unfilled by design. A real
hole or adjacent shadow with matching erroneous color and depth can still be
filled within the bound. Colored shadows and color registration errors can create
false seeds anywhere; bounding growth does not remove false seeds. Do not claim
universal contour improvement, topology preservation, or real-image accuracy.
Compare added and rejected pixels on independently annotated, registered real
pairs before selecting this mode or changing its parameters.

## Settings and debug

Constructor defaults and valid ranges are listed in the
[tuning guide](RGBD_PICK_TUNING.md#segmenter-settings).

Check/compensation reuse the existing color-reference implementation. A rejected
check disables color evidence and keeps usable depth in legacy modes; evidence
mode requires strong, supported depth without color confirmation. Color_seeded
mode has no seeds and defers depth candidates when color is rejected. It never falls back to
border colors. Too few check pixels is an input error. These checks concern color
only; depth reference noise/drift is not compensated or validated by that region.

Result masks include `object_mask`, `uncertain_mask`, `observed_mask`,
`color_foreground_mask`, `depth_foreground_mask`, `depth_valid_mask`, and
`conflict_mask`, `suppressed_mask`, and `deferred_mask`. `observed_mask` describes availability under the fusion policy,
not semantic certainty and not raw current-depth validity alone. Pass it to
`OuterContourDetector` so unavailable pixels and ROI cuts are reported as such;
in evidence/color_seeded/depth_anchored modes prefer `outline_observed_mask`.
Depth_anchored additionally excludes nonselected unknown pixels; see its guide.
Do not use `~uncertain_mask` as the outline observation domain by default: that
would cut out useful color-only objects and create artificial contours. Propagate
segmentation uncertainty separately when interpreting outlines. For metric outline
lengths, supply current-depth validity and camera intrinsics under that detector's
normal contract; do not substitute the pair-validity mask without considering it.

The fusion stage reports `candidate` when selected pixels exist,
`no_change_candidate` when observations exist but none pass selection,
`insufficient` when the ROI has no usable observations under the policy, and
`empty_domain` when the ROI itself is empty. In evidence mode, `unresolved` takes
precedence over `no_change_candidate` when no pixels are selected but suppressed
or deferred candidates remain (with usable observations). Counts/masks must still
be checked when status is `candidate`: other regions can remain unresolved.

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

## Dark shadows and much nearer objects

For a fixed registered empty/current RGB-D pair, existing APIs can conservatively
select **visible raised target candidates within a declared camera-Z band**.
No new segmentation mode is required. Existing defaults remain unchanged.
Run `examples/example_rgbd_shadow_occluders.py` for a small ndarray example.

### Fixed settings across captures

The 5/80 mm example is not a universal preset. One fixed upper bound requires a
noise-separated gap between the largest legitimate target and the nearest unwanted
surface over the entire operating range. Object lifting, camera tilt and depth
reference drift can remove that gap. See [fixed-setting conditions](RGBD_PICK_TUNING.md#fixed-settings-across-captures) before selecting deployment values.

### Which evidence to require

- `fusion_mode='depth_required'` selects only valid empty-current camera-Z
  differences greater than `depth_threshold_m`. A shadow whose depth equals the
  table therefore cannot pass, even if it is extremely dark or colored.
  This mode means depth only, not RGB AND depth.
- Exclude valid nearer surfaces with a caller-derived `work_roi`. Nearer surfaces can pass
  foreground selection; no mode semantically identifies an occluder.
- Use `missing_depth_policy='unknown'` for any color-assisted mode. The default
  `'color'` can select invalid-depth occluders from their RGB. The depth-required
  mode already requires valid depth regardless of this policy; spelling it out
  keeps the intended policy when switching modes.
- To preserve the earlier `color_seeded` restriction, intersect its mask with
  the depth-required mask, with the same ROI and missingness policy. Do not union
  it back in: strict chroma seeds may have valid depth equal to the table.

`color_seeded` and `evidence` alone are not dark-shadow rejection guarantees.
Even uniform RGB attenuation on a colored table can change encoded-Lab chroma
sufficiently to form a seed. The existing boundary detector's
[`uniform_color` shadow hypotheses](BOUNDARY_SHADOW.md) are useful only when their
brightness, color-model and seed assumptions hold. Very dark/clipped pixels do
not have reliable normalized RGB ratios. Lowering `shadow_min_ratio` alone does
not make these pixels safe, and object/shadow identical appearance is ambiguous.

### Declare the usable depth band

For millimetre inputs:

```python
from depth_image_lens import RGBDBackgroundSegmenter, background_difference

delta_mm, pair_valid = background_difference(
    depth_mm, empty_depth_mm, depth_unit='mm', background_unit='mm',
    invalid_values=(65535,))
near_excluded = pair_valid & (delta_mm > 80.)
roi = application_roi & ~near_excluded
result = RGBDBackgroundSegmenter(
    fusion_mode='depth_required', missing_depth_policy='unknown',
    depth_threshold_m=.005,
).detect(color, depth_mm, background_color=empty_color,
         background_depth=empty_depth_mm, color_order='RGB',
         registered_to_depth=True, depth_scale_m=.001,
         invalid_values=(65535,), work_roi=roi, diagnostics=True)
```

Here the admission band is `5 mm < empty-current Z <= 80 mm`. These are example
application bounds, not fitted defaults or physical height estimates. Choose a
maximum that contains the legitimate target and is separated from the unwanted
near object by a measured noise margin. Keep metric units explicit. Never subtract
unsigned arrays directly. Preserve original sensor masks and declared sentinels;
`0`, negative, nonfinite and declared sentinel samples remain unknown. If the
sensor provides extra validity masks, use them consistently both when deriving
the exclusion mask and in `detect`.

An absolute camera-Z near limit can instead use valid **current** depth, for
example `current_valid & (depth_m < declared_min_z_m)`. Build `current_valid` from
sensor validity, positive finite depth and sentinels, not color or a segmented
mask. Do not use returned `result.depth_valid_mask` as raw sensor validity: it is
already intersected with background-depth validity and the ROI. For tilted-camera
physical table height, use calibrated `CameraIntrinsics.deproject` and
`TablePlane.height` with a verified current table plane; camera-Z difference is
not table-normal height.

### Preserve unknown regions and diagnostics

Keep `near_excluded` and `~pair_valid` separately. Both are outside the downstream
observed target domain, but only the first has a usable measured near surface.
Missing depth does not establish what was present. Masking the near surface cannot
restore the target hidden behind it.

Pass `result.outline_observed_mask` into `OuterContourDetector.detect`. An outline
next to an observation/ROI cut is clipped or uncertain. Do not replace observation
with an all-true mask or fill the cut. For the optional color/depth intersection,
intersect both observation masks and exclude disagreement pixels; retain the
union of uncertainty masks plus disagreement separately for pick checks.

`outline.holes` describes binary mask topology only. An enclosed missing-depth
patch can appear there without per-hole visibility metadata; it is not a verified
physical hole. No hidden perimeter, full target area or safe grasp is inferred.

Useful arrays are the external exclusion/validity/ROI masks and the segmenter's
`depth_difference_m`, `depth_threshold_margin_mask`, `object_mask`,
`outline_observed_mask` and `uncertain_mask`. No new diagnostics API is needed.

### Necessary tradeoffs and counterexamples

- A true target thinner than the lower threshold or with missing depth is lost
  or left unknown, even when color supports it. Weakening the depth requirement
  to recover it can also admit indistinguishable table shadows.
- A legitimate raised part beyond the declared maximum is also excluded.
- A second object inside the admitted band is not identifiable as unwanted from
  depth alone. Its RGB may also match the target. Use an independently justified
  ROI, semantic exclusion, tracking or another observation when needed.
- Coherent depth bias larger than the lower threshold can make a shadow look like
  a raised surface. Thresholding cannot distinguish identical RGB-D observations.
- Color intersection can suppress depth artifacts, but loses same-table-color or
  unseeded true objects. Depth-only remains available for these cases.
- Registration errors, mixed boundary depths, sensor noise and wrong reference
  depth can still corrupt a silhouette. These synthetic regressions establish
  contracts and explicit failure cases, not real-image accuracy. Validate actual
  registered pairs before choosing deployment thresholds.

## Color-supported local growth

For gray objects with supported depth but weak color difference, and rejection
of unsupported color-only shadows, see [color_supported](RGBD_COLOR_SUPPORTED.md).
This opt-in mode retains per-seed colors and bounds expansion; it does not alter
color_seeded or the default union rule. Missing depth remains unknown.
