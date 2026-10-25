# Independent visual grasp mode

`IndependentGraspDetector` estimates each image-side hand independently from
separated hand and deformable object regions. It does not call the bridge/sag
detector or require a spanning object region. The existing full detector remains
available. No training, model service, new dependency or background image is needed.

```python
from depth_image_lens import IndependentGraspDetector

result = IndependentGraspDetector().detect(
    depth, object_mask, {"left": left_hand_mask, "right": right_hand_mask},
    depth_unit="m", registered_to_depth=True,
    hand_visibility=("partial", "visible"),
    mask_complete=(False, True), diagnostics=True,
)
print(result.sides["left"]["estimate"])
print(result.sides["right"]["stages"])
```

Depth is same-grid HxW optical-axis Z in explicit `m` or `mm`. Masks are bool
arrays. `detect_16uc1(..., depth_scale_m=.001)` accepts uint16 counts with a
configurable metres/count scale. All inputs must already be registered. Zero,
negative and nonfinite depth are invalid; `observed_mask=False` separately
withholds measurements. No depth is filled or interpolated.

Hand masks must describe the end effectors, not entire arms. Assign image-left
and image-right masks explicitly; these labels never determine physical robot
identity. Overlap between semantic masks and optional `uncertain_mask` pixels is
excluded and diagnosed instead of silently assigned to an object or hand.

Each side returns `estimate=likely/unlikely/unknown`, reasons, warnings and three
stages: boundary proximity, depth consistency and grasp inference. Numerical
stages are `computed` when measurable, while the grasp hypothesis is `inferred`.
`physical_grasp` remains `unknown`: a nearby surface, crossing, passive contact or
segmentation error can resemble a held deformable object. Static RGB-D proximity
cannot establish mechanical holding. Interpret `unlikely` as absence of the
configured visible contact evidence under the declared completeness assumptions.

The algorithm searches object boundaries within `contact_radius_px=3` of each
hand-boundary pixel. It uses nearest image-distance ties, so a farther matching
surface cannot bypass a closer depth mismatch. Adjacent centers are one pixel
apart; the radius permits limited image-space separation, not an invented contact
or a filled gap. All accepted pairs retain actual measured depth. The default
`depth_tolerance_m=.04` checks absolute camera-Z difference, not Euclidean 3D
contact distance. Pixel radius must be chosen for the processing resolution.

A likely result requires at least `min_contact_pixels=3` in a four-connected
consistent boundary region on **both** masks within the same corresponding
contact region, plus consistent evidence on at
least `min_consistent_fraction=.6` of near hand pixels with depth **within one
corresponding contact patch**. One matching
object pixel or disconnected speckles cannot meet the default support rule.
Matched depth-consistent pairs are partitioned by their hand/object four-neighbor
components, repeating within subsets until both endpoint sets are contiguous.
Support from unrelated regions is never combined to pass the two-sided minimum.
`corresponding_contact_regions` records each paired region and its support; the
separate largest-component fields remain raw measurements, not a joint proof.
Contact patches are partitioned from all nearest boundary pairs before depth
validity/consistency filtering. Missing and inconsistent neighbors in the same
patch remain visible in its measurements. Each patch must independently have a
sufficient connected matched region and consistent fraction; disconnected small
supports cannot be pooled. One qualifying patch is enough for the side's local
positive inference. A separate inconsistent or missing patch cannot dilute it.
A continuous mixed patch still uses its full valid evidence: it is not split by
depth agreement merely to force a positive. Global consistent fraction and
coverage remain diagnostics, not a veto on another qualifying patch.
`depth_consistency.contact_patches` records patch IDs, bounding boxes, support,
coverage, consistent fraction, positive status and failure reason. Patch IDs are
per-call geometry identifiers, not tracked hands or physical contacts.
Partial visibility and missing depth elsewhere do not veto a supported local
hypothesis. Measurements and warnings remain available even if a rule fails.

An unlikely result requires no near boundary, or no depth-consistent pair with
sufficient observed coverage (`min_valid_fraction=.8`). It also requires the
side's `mask_complete=True` attestation, a fully visible hand away from the frame
edge, and no nearby ambiguous labels. Completeness means the hand and relevant
object search neighborhood are represented; it cannot be inferred from a final
mask. The default `(False, False)` retains negative measurements without asserting
absence. A hidden/empty hand region stays unknown while the opposite side runs.
A hidden declaration with nonempty hand pixels warns and retains the measurements
without treating those pixels as verified visible evidence.

`result.diagnostics` contains per-side boundaries, near pixels, consistent hand
and object pixels, and nearest valid pair coordinates/depth deltas; global arrays
separate sensor invalidity, caller unobserved pixels and ambiguous labels.
`to_dict()` returns only JSON-safe scalar summaries, not these arrays.

## Optional RGB candidates

`detector.detect_color_candidates(depth, colors, hand_regions, depth_unit="m")`
reuses a registered `SeededColorAssistant` result. `hand_regions` contains two
caller-assigned bool regions restricting the hand appearance class to each end
effector. Both appearance classes must be available for separated masks; a
single object-color similarity hypothesis does not establish hand identity.
The helper preserves appearance uncertainty, including missing classes, and
labels its segmentation source. Eroded color boundaries can increase image gaps;
inspect them rather than widening the radius to force a result. The helper does
not locate hands, verify alignment or map image sides to robot IDs.

Run `python examples/example_grasp.py` for a small generated example, and
`python -m unittest tests.test_grasp -v` for focused regressions. These tests cover
explicit geometry contracts; they do not establish field accuracy. Real-data
annotations, evaluation scripts and galleries remain outside this MIT repository.

## Tuning local versus global evidence

All five thresholds are constructor arguments. `contact_radius_px` is an integer
from 1 to 8; other support and fraction constraints are validated at construction.
Inspect registration/masks and processing resolution first, then image radius,
metric depth tolerance, local connected support and local consistent fraction.
`min_valid_fraction` remains the coverage gate for depth-based negative inference;
it does not discard a sufficient observed local positive patch. Whole-side
quality warnings remain present with a local positive. Negative completeness and
visibility rules are unchanged. Do not interpret a local positive as whole-object
coverage or verified physical holding.
