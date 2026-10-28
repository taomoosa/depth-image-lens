# Independent background color and brightness tolerances

`OverheadBoundaryDetector` can compare a registered empty color reference without
an object-color seed. Enable this optional mode explicitly; the default
`background_difference_mode="lab_distance"` preserves legacy selection.

```python
from depth_image_lens import OverheadBoundaryDetector

# current / empty: aligned uint8 HxWx3 RGB, same camera, crop and table.
model = OverheadBoundaryDetector(
    background_difference_mode="separate_tolerances",
    background_chroma_tolerance=30.,
    background_luma_tolerance=60.,
)
result = model.detect(current, color_order="RGB", background_color=empty,
                      diagnostics=True)
foreground = result.object_mask
unknown = result.uncertain_mask
print(result.stages["background_difference"])
result.save_debug("background-tolerances.png", current)  # explicit, no overwrite
```

Depth is optional and does not determine color selection. No object seed or full
object mask is required. When possible, pass `background_check_mask`, a bool HxW
patch of unchanged visible table in both images. Without it the reference is
unchecked. Passing a check is not proof that the entire reference is empty.

## Exact rule and units

After the existing optional global Lab offset correction:

```
dL = abs(current_L - corrected_reference_L)
dC = hypot(current_a - corrected_reference_a,
           current_b - corrected_reference_b)
S  = (dC / Tc)**2 + (dL / Tl)**2
foreground candidate = available_reference AND S > 1
```

`Tc=background_chroma_tolerance` and `Tl=background_luma_tolerance` use
**OpenCV's encoded 8-bit Lab units**, not standardized Delta E, radiance, metres or
calibrated confidence. Both default to 30. They must be finite positive numbers.
This is an elliptical distance test, not independent AND/OR tests: two normalized
residuals of 0.8 give S=1.28 and select foreground. S=1 is not foreground.
Existing minimum-component-area filtering follows selection, so tiny candidate
components can still disappear. No morphology is added by this mode.

With Tc=Tl=T, the foreground decision is the same as the legacy Euclidean encoded
Lab distance >T, including equality. The implementation uses that norm directly
for the equal-tolerance decision. The existing `color_distance` controls legacy
and fallback segmentation, but does not replace Tc/Tl in this mode.

## Uncertainty and diagnostics

`background_score_margin=8/30` is a dimensionless **radius** margin:
`abs(sqrt(S)-1) <= margin` is uncertain. It is not a squared-score margin. Values
from 0 to 1 are accepted; zero retains equality as uncertain. At equal tolerances
30 this corresponds to the legacy 8-unit band, except for clipping checks below.
At other tolerances it scales with the ellipse; changing `color_margin` does not
change this band. Uncertainty may overlap retained foreground.

Pixels where either original RGB image has any channel <=1 or >=254 are marked
uncertain in this mode because dark clipping/saturation loses information. The
foreground decision is retained; this is not an automatic shadow label. Invalid or
uncovered reference pixels remain unknown, cannot be foreground and have NaN in
numeric diagnostic arrays. A low score is agreement under the configured model,
not proof of empty background. Same-color objects can be completely invisible.

With `diagnostics=True`, float HxW maps are:

- `background_chroma_distance`: dC, encoded-Lab chromatic distance.
- `background_luma_distance`: dL, encoded-Lab luminance distance.
- `background_combined_score`: S, dimensionless; threshold 1.

Bool HxW maps also include `background_score_saturated_mask` and
`background_photometric_clipped_mask`. To handle extremely small positive
settings, S saturates at 1e12; this preserves its >1 decision and is explicitly
flagged. No Inf is emitted on available pixels. A saturated value is a lower bound,
not its true magnitude. `to_dict()` remains strict-JSON safe. Render/save debug
shows separate dC/dL/score heatmaps; the score image clips its display at 4 while
stored measurements remain available. With diagnostics disabled, result masks,
parameters and stage summaries remain available, but these maps are absent.

## Interaction with existing features

- Alignment, reference validity, integer `background_shift_xy` and optional
  `background_compensate=True` run first. Compensation is the existing single
  median additive encoded-Lab offset, not an arbitrary spatial illumination model.
- `background_check_p95=20` and `background_max_offset=40` remain **unweighted
  encoded-Lab** checks. Increasing Tl does not relax them. A rejected reference
  still uses the documented uncertain border fallback; its score maps are only
  diagnostics. Check `applied_to_selection` before interpreting them as the result.
- `background_shadow_mode="off"` remains default. If the separate seeded
  `"uniform_color"` shadow mode is also enabled, it runs **after** the ellipse on
  its foreground candidates and still needs a distinct small object-color seed.
  Neither feature invents foreground discarded by the other. They answer different
  questions: adjustable difference sensitivity versus conditional shadow removal.
- Explicit `object_mask`, including an empty mask, overrides background selection.
  Optional bad references or missing depth preserve existing partial-result rules.
  Constructor settings are validated even if the selected call does not use them;
  an invalid configuration is distinct from unavailable optional image evidence.

## Tuning and limitations

Start with equal tolerances to reproduce the old decision, inspect dC/dL separately,
and raise Tl modestly when measured illumination changes mostly affect brightness.
This also removes objects distinguished only by brightness. Raising Tc tolerates
chromatic changes but can hide colored objects. Adjust one tolerance at a time and
check object retention, background/shadow false positives and unknown coverage;
there is no universal best pair. 30/60 is an illustration, not a validated preset.
The small normalized margin only describes proximity to the decision boundary;
it is not learned or calibrated uncertainty about physical identity.

A shadow and a dark object can have identical RGB, so thresholds cannot guarantee
separation even when their unshadowed material colors differ. Colored shadows,
texture, registration error and deep darkness remain limitations. Existing real
comparison references were reconstructed with target-mask assistance: their
results contain target-information leakage and do not establish independent
paired-background accuracy. Acquire genuine empty/current captures and appropriate
object/shadow labels for site-specific calibration. No score proves a wrinkle,
layer count, physical contact or grasp.
