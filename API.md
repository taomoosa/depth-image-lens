# API and input contract

All generic scene targets are called **deformable objects**. The existing identifiers `work_roi`, `work_mask`, `work_pixels` and related reason strings are retained for API compatibility; they refer to the deformable-object search region, not a separate object category.

## Public API

```python
from depth_image_lens import BridgeDrapeDetector, BridgeDrapeResult

BridgeDrapeDetector(
    *, threshold_m=0.02, min_area=20, anchor_band_px=3,
    max_gap_px=0, sag_ratio=0.15, drape_ratio=0.25,
)

# Manual anchors:
detector.detect(
    depth, difference, hand_rois, *,
    depth_unit="m", difference_unit="m",
    difference_sign="background-current",
    background_valid=None, observed_mask=None,
    arm_mask=None, work_roi=None,
    hand_visibility=("visible", "visible"), down=(0, 1),
)

# Shape-derived anchor candidates:
detector.detect_auto(
    depth, difference, *, upper_band=None, min_span_px=30,
    max_height_delta_px=12, depth_jump_m=0.10, down=(0, 1),
    **options,
)
```

The signatures above describe the API, not executable calls with placeholder variables. `detect_auto` accepts unit, difference-sign, validity, arm-mask and search-ROI options from `detect`; it rejects manual hand ROIs and hand-visibility declarations. See [AUTO.md](AUTO.md).

## Assumptions and inputs

| Input | Contract |
|---|---|
| `depth`, `difference` | Numeric HxW arrays with identical pixel coordinates, shape, unit and depth meaning. A fixed camera and unchanged background are required. |
| Units | Explicit matching `m` or `mm`. Threshold settings ending in `_m` always use metres. No automatic unit inference. |
| Difference | Default `background-current`: positive values are nearer than background. `current-background` is also accepted and sign-corrected. Negative differences do not become foreground. |
| Invalid data | Current depth <=0 or nonfinite is invalid. Difference must be finite and reconstructed background depth positive. Invalid background must be NaN in the difference or false in `background_valid`. Zero-filled/clipped differences cannot recover missing provenance. |
| Masks | `background_valid`, `observed_mask`, `arm_mask`, `work_roi`: bool arrays of the same shape. Arm-mask true excludes a pixel. Search-ROI true includes a pixel. Missing/excluded pixels are not known empty space. |
| `hand_rois` | `{"left": (x0,y0,x1,y1), "right": (...)}`; x1/y1 exclusive. Lower edges should be near the candidate contact. A thin anchor band is retained while hand interiors are excluded. |
| `hand_visibility` | Two caller declarations: `visible`, `partial` or `hidden`. Valid depth alone does not establish unobstructed visibility. Mark known occlusion in `observed_mask` too. |
| `down` | Finite nonzero image vector, normalized internally. Default `(0,1)`. Anchor heights may differ in the image. Left/right are image-relative IDs, never automatically robot-hand IDs. |

This mode assumes a static deformable object suspended from above, with approximately equal hand heights for automatic candidates. It does not support arbitrary camera/background changes or infer physical contact from a closed gripper.

## Results

| Output | Meaning |
|---|---|
| `bridge` | `yes/no/unknown`: an observed candidate path reaches both anchors, no path exists in a complete visible search, or evidence is insufficient. |
| `large_sag` | `yes/no/unknown`: robust upper-envelope deviation from the sloping anchor chord, normalized by anchor span. It does not measure the lower edge or physical tension. |
| `drape` | `left/right/both/neither/unknown`: anchor-connected downward extent exceeds the configured span ratio. A one-sided result requires observable opposite anchor/search conditions. |
| `features` | Sag ratio, span coverage/length, per-anchor hanging extent/width, validity and gap information. Image lengths are pixels; ratios are dimensionless; unavailable values are `None`. |
| Masks | `foreground_mask`: valid positive differences. `work_mask`: candidate region after search/exclusions/area filtering. `invalid_mask`: missing depth/difference, not empty space. |
| `auto` | Present only for automatic candidates: stage, candidate ROIs/points, opposite search region and an explicit unverified-contact flag. |
| `validation_status` | `not_validated`: real accuracy has not been established; this is separate from an observation's `unknown` result. |

`result.to_dict()` produces JSON-safe values. `include_masks=True` includes bool lists and can be large. Individual outputs have independent conditions: for example, `bridge=no` can coexist with `large_sag=unknown`.

`drape=both` means enough extent belongs to components connected to each anchor. It does not imply two separate hanging pieces or symmetric local shape. A horizontal bridge with a long lower edge can have `drape=both` and `large_sag=no`.

## Tuning order

1. Verify registration, units, sign and invalid-data provenance before tuning thresholds.
2. Set manual hand boxes or the automatic upper band. Preserve the narrow anchor/contact band when excluding hands/arms.
3. Exclude upper arm paths with `arm_mask`; restrict the candidate region with `work_roi` when needed.
4. Set `threshold_m` from static-background noise. The default 20 mm is an example, not a universal validated value.
5. Set `min_area` and `anchor_band_px` for the image resolution. Each anchor requires at least three connected pixels.
6. Prefer `max_gap_px=0`. Optional 1-2 pixel jumps cross only observed pixels; they never fill invalid/excluded regions or large holes. Inspect `gap_used`.
7. Adjust `sag_ratio` and `drape_ratio` using representative labeled scenes. Sag requires at least 95% observed span coverage. Apparent notches, openings, overlap and arm contamination can confound the upper envelope.

Connectivity is eight-neighbour. Visible paths around small holes can support a bridge; missing regions are never fabricated. Downward clipping can hide paths and yield unknown. Python component traversal can be slow on large images; crop consistently, updating every ROI/mask coordinate. No throughput guarantee is made.

## Real-scene validation still required

Record one fixed-camera empty background and static states showing a horizontal bridge, deep U, left-only drape, right-only drape, separate drapes and arms alone. Independently check masks, anchor candidates and actual contacts. There are currently no matched real background pairs in this project's tests, and no synthetic background is reported as a real evaluation.
