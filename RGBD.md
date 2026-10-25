# Optional oblique RGB-D color assistance

Color can help separate appearance when depth foreground contains both hands and
a deformable object. Cropping color by depth alone does **not** segment them.
`SeededColorAssistant` is a small, optional NumPy-only baseline, independent of
skin color. Foreground and unit conventions are unchanged. The feature-wise behavior below
applies to depth geometry as well as RGB-D. No pretrained
model, network service, training or additional runtime dependency is used.

## Method choice

| Approach | Benefit | Limitation and decision |
|---|---|---|
| Foreground-constrained seeded RGB distance | Small, inspectable; works with arbitrary hand colors and several seeded object colors | No spatial/material semantics; selected baseline with explicit unknown pixels |
| Seeded region growth using depth and color | Can limit disconnected color matches | Can leak through similar touching surfaces and split folds; not added in this pass |
| GrabCut / color mixture plus graph cut | Combines color distributions and boundaries, supports user strokes | Iterative and may require corrections; extra OpenCV dependency and hard assignments do not establish contact; researched, not implemented here |

The [original GrabCut work](https://www.microsoft.com/en-us/research/publication/grabcut-interactive-foreground-extraction-using-iterated-graph-cuts/)
and [OpenCV tutorial](https://docs.opencv.org/4.13.0/dd/dfc/tutorial_js_grabcut.html)
explain the graph-cut alternative. This implementation is our own simple color
baseline, not a reproduction of that algorithm or its published accuracy.

## Inputs and outputs

```python
from depth_image_lens import SeededColorAssistant

colors = SeededColorAssistant().segment_16uc1(
    depth, difference, color,
    object_seeds=object_seeds, hand_seeds=hand_seeds,
    color_order="RGB", registered_to_depth=True,
    depth_scale_m=0.001, threshold_m=0.02,
)
```

- `depth`, `difference`: same-grid HxW NumPy uint16. Difference is clipped
  `max(empty_background-current, 0)` with irrelevant regions set to zero.
  Zero difference is non-foreground, including at threshold zero; raw depth zero
  is invalid. Scale is explicitly metres/count; 0.001 is the convenience default.
- `color`: aligned uint8 HxWx3 with explicit `RGB` or `BGR`. Same dimensions alone
  do not prove registration. The caller must supply synchronized, consistently
  rectified/cropped/resized data and declare `registered_to_depth=True`.
  See [RealSense alignment and occlusion guidance](https://dev.realsenseai.com/docs/projection-texture-mapping-and-occlusion-with-intel-realsense-depth-cameras/).
- `object_seeds`, `hand_seeds`: disjoint bool HxW masks, manually selected or
  provided by the caller. Small interior strokes or ROIs are suitable. Include
  relevant object patterns and hand/arm appearances. This helper does not find
  hands automatically. Seed semantics remain caller assumptions, not truth.
- Optional `observed_mask`, `background_valid`, `color_valid_mask`: bool HxW.
  Missing/occluded/registration-uncertain pixels must be marked false. Foreground
  cannot be recovered from missing depth. Seeds outside available foreground are
  ignored and counted. Other raw sentinels require caller validity masks.
- `object_mask`, `hand_mask`: disjoint **appearance candidates** within available
  depth foreground. `hand_mask` also represents the seeded arm appearance class.
  `uncertain_mask` includes undecided foreground and unavailable depth pixels.
  `appearance_uncertain_mask` separates genuine class ambiguity from the extra
  erosion-only boundary guard. `hand_exclusion_mask` is the separated hand
  appearance before erosion, suitable only as a candidate exclusion constraint.
  None of these masks turns uncertain appearance into a new sensor measurement.
  `to_dict()` returns a strict-JSON summary, counts, coverage and reasons; it does
  not serialize large arrays. No calibrated probability or physical holding claim.

For existing float depth workflows, `segment(color, foreground_mask, ...)` accepts
caller-generated bool foreground and `valid_mask`, without unit conversion.
Do not replace missing real background with an invented background and describe
that as measured subtraction.

## Parameters and integration

Color distance is Euclidean distance in 8-bit RGB, not perceptually uniform or
illumination invariant. Nearest colors are drawn from each seed class (at most
64 deterministic distinct-color samples). A class is accepted only within
`max_color_distance=60` and ahead of the other class by
`min_color_margin=15`. Same/unknown colors abstain even at seed pixels. A bounded
palette may miss rare appearances; these values are experimental settings.

`boundary_guard_px=1` erodes each candidate class by one pixel, marking uncertain
boundaries rather than filling holes. Values 0..3 are supported. Zero is an
explicit no-buffer assumption, **not** evidence of correct registration; a guard
cannot guarantee protection from an incorrectly registered image. Never relax it
only to force a preferred bridge result.

1. Check alignment, units and original validity first; inspect seed labels.
2. Inspect candidate/uncertain masks. Add representative interior color seeds
   before widening color thresholds. Similar colors may remain unresolvable.
3. Keep the depth-only result separately labeled. The default example now keeps
   original depth validity, uses the full separated hand appearance as an
   exclusion outside thin candidate anchor bands, and separately checks semantic
   uncertainty on the depth path. Run `examples/example_rgbd.py`; its `run(...)`
   accepts ndarray inputs and optional original validity/arm masks.

The small `apply_color_constraints(...)` function in that example is deliberately
experimental, not a new detector class. Both eroded appearance interiors are required for semantic classification.
When a class is absent, that classification abstains, while independent geometry
on depth and color candidates remains in `stage_results`. Existing
caller arm exclusions are never cleared at anchors. The full hand appearance is
excluded because eroding an arm exclusion could leave its outline as an artificial
bridge. Exclusion does not mark that region as observed empty space.

After testing depth connectivity, a genuine appearance-unknown region on its
corridor (outside candidate anchor bands) makes the object bridge/sag classification
unknown. The same check applies to drape components. Geometry before this semantic
gate is retained in `features["rgb_depth_geometry_before_semantic_gate"]`; it is
not a confirmed object path. Erosion-only uncertainty does not invalidate an
otherwise observed depth pixel. No hole or zero-difference gap is filled.

Use the **same scale and foreground threshold** in segmentation and detection.
Original `observed_mask` and `background_valid` go unchanged into both detector
calls. Color validity belongs to segmentation; it is not depth validity. Combine
caller arm masks by OR, and never restore unavailable pixels. Retaining a thin
anchor band preserves the existing geometrical candidate convention; it does
not confirm hand/object contact. Automatic mode accepts exclusion masks, but
this example/comparison uses manual coarse ROIs, not automatic hand recognition.

Hidden supports and physical hand identity retain the limitations in AUTO.md.

Run the standalone in-memory example and unit tests:

```bash
python examples/example_rgbd.py
python -m unittest tests.test_color -v
```

## Optional observed support and classified routes

Manual detection accepts `anchor_policy="local_observed"`, requiring an actually
observed 3x3 foreground patch at each anchor in the same depth-continuous component.
It requires `max_gap_px=0` and fills no holes. The independent feature pass now
uses visible band support even when the conservative complete-hand check fails;
negative claims retain their completeness requirements. The in-memory RGB-D
example exposes this as `local_anchor_support=True`.

`examples.example_rgbd.run(..., semantic_path_policy="observed_route")` can exclude
appearance-unknown pixels from a separate positive route check. A route must still
reach both anchors using classified observed depth. It never turns a failed route
check into a confirmed negative. Default component gating remains unchanged;
drape uses the original component gate and sag uses verified route geometry.
`rgb_component_before_route_mask` retains the original component, while
`bridge_corridor_mask` is the verified route only after successful verification.
`rgb_semantic_route_exclusion_mask` and `rgb_semantic_route_used` distinguish this
check from raw observed foreground. No semantic or sensor uncertainty is filled.

For masked differences, `guard_masked_negative_claims` in
`depth_image_lens.rgbd_diagnostics` is a separate opt-in provenance check. It takes
a diagnostics-enabled result, uint16 difference and optional bool
`verified_nonforeground_mask`. This mask must attest measured background or
independently confirmed irrelevant exclusions, not be recreated from final zeros.
Unverified zero pixels in the search region withhold bridge `no` and drape
`left`/`right`/`neither`; observed positives remain. No arrays are altered, no path
is recovered, and incorrect caller attestations cannot be discovered from the
masked input. Default zero-difference validity remains unchanged.

Detailed benchmarks, dataset eligibility checks and review galleries are local
external records. This package does not depend on evaluation scripts or datasets.

## Supplied object masks without a measured background

`BridgeDrapeDetector.detect_mask` accepts an explicitly supplied boolean object
mask, metric current depth, and optional manual candidate hand ROIs. It shares the
existing geometry implementation; it does not synthesize a difference image or an
empty background. This is an alternative input experiment, not validation of
background subtraction, automatic segmentation, hand detection, or physical grasp.

```python
result = detector.detect_mask(
    depth, object_mask, hand_rois, depth_unit="m",
    arm_mask=robot_exclusions, observed_mask=observed,
    anchor_policy="local_observed", mask_complete=False, diagnostics=True,
)
```

True mask pixels are caller-supplied object evidence. False mask pixels are
excluded. Nonpositive/nonfinite depth is sensor-invalid, while `observed_mask`
can separately withhold uncertain measurements. Diagnostics retain these three
categories. No hole interpolation or gap inference is allowed by this API.
Depth units must be explicitly `m` or `mm`. With `hand_rois=None`, observed bridge
remains unknown, while silhouette, visible-chord sag and static support hypotheses
continue independently. Use the same down-side hand-box convention as `detect`.

The default `mask_complete=False` withholds negative bridge and one-sided/neither
drape claims. It retains observed bridge paths, hanging extents and measured
sag relative to candidate bands, with explicit mask-completeness warnings. `mask_complete=True` is a caller attestation
that mask exclusions are verified non-object/irrelevant throughout the search;
it must not be enabled merely to obtain a decision. Candidate anchors and a
spanning mask still do not verify contact. Pre-guard geometry is recorded in
`features.supplied_mask_geometry_before_guard` for debugging, not as the final
prediction. Raw depth zero remains invalid; difference zero remains excluded.

Keep real-data masks, images, license evidence, and review galleries outside this
MIT repository. Preserve annotation provenance, distinguish provisional masks from
independent ground truth, and report unscored predictions as such. A confidence
filter comparison is a separate condition, not proof that its positive outputs
are correct. Existing small real batches may be insufficient for an accuracy or
false-bridge estimate even if a geometry candidate is found.


## Independent features and evidence quality

`stage_results` exposes each feature with `computed`, `inferred`, or
`insufficient_input` status. A computed value can still be `unknown`: the
measurement ran but did not identify a positive or negative claim. Quality
warnings never imply that a different feature was unprocessable.

| Feature | Required evidence | What failure leaves available |
|---|---|---|
| `observed_bridge` | Two caller attachment bands joined by observed depth, minimum area and local support | Other components, per-band extents, silhouette measurements |
| `attachment_drape` | Usable depth connected to each band independently | The other band and inferred contour reference |
| `morphology` | Eligible supplied silhouette, no depth required | Width, extent and crop-boundary status even with missing depth |
| `hanging` | Silhouette extent and chosen image-down direction | Visible morphology; physical hanging uses the static assumption |
| `sag` | Sufficient visible transverse span and a reference chord | Numeric contour ratio when actual support chord is unknown |
| `support_hypotheses` | Silhouette plus `static_hanging=True` | One/two-support alternatives, upward rays, no physical hand identity |
| `grasp` | Physical evidence beyond static shape | Always unknown in this detector |

Depth discontinuities cut individual neighbor transitions; an alternative path
must use actual adjacent observed depth. No gap, missing depth or excluded pixel
is filled. A partial/outside hand ROI warns but does not block a path through its
visible attachment band. At least three foreground band pixels are required,
plus either complete band observation or a valid 3x3 local patch in each band.
Overlapping bands and spans below eight processing pixels cannot establish a
pair. Invalid array contracts still raise errors.

Root `bridge=yes` now means observed visual connection between candidate bands,
including partial hands, rather than passing every conservative visibility check.
Root sag reports the visible envelope relative to that candidate chord, even
when the mask is incomplete; inspect warnings before interpreting it as the true
object envelope. Historical gated decisions remain in
`features.conservative_decisions`. Diagnostic `feature_*` masks describe the
independent pass; legacy rejected-component masks remain distinct.

Without observed bands, sag uses visible contour endpoints as an **inferred**
reference. Its numeric ratio is retained, but top-clipped silhouettes cannot
identify actual support sag. Support-count hints use upper-entry lobes/neck shape
and observed band connectivity. They always retain `[1, 2]` alternatives:
several hidden supports can share one visible neck, or one support can create two
visible folds. Rays indicate above/diagonal-above directions with unknown distance;
they are not detected offscreen hands. Set `static_hanging=False` to disable
physical support hypotheses. This is a static experiment, not motion estimation.

`analyze_color_geometry(detector, depth, colors, depth_unit="m")` is the reusable
candidate-geometry helper. Missing hand seeds may still supply a single-class
object-color similarity hypothesis through `object_appearance_candidate_mask`.
That mask does not establish object/hand identity. Similar colors remain
ambiguous. The example keeps color-candidate and depth-candidate stages separately
from its conservative semantic classification.

Related dependency audit: Lift now computes finite height samples below its
coverage threshold while its lift decision remains unknown; height failure does
not erase other finite samples. Corner keeps polygon acceptance separate from
local line and depth-patch measurements. Low depth coverage can retain a plane
residual but cannot pass the existing candidate-quality checks. Shapes with no
polygon candidate, non-deprojectable height samples, or fewer than three
nondegenerate plane points remain unprocessable for those specific features.
These additions do not validate grasp, contact, single layers, or whole-object
visibility. Real-data evaluation remains external to this repository.
