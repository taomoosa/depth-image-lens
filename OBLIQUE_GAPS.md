# Missing-depth stripes in oblique views

An invalid depth stripe can split one cloth silhouette into two measured
components. Lowering a positive-depth threshold does not restore the missing
pixels. Nor do `depth_jump_m`, `min_width_px`, or `min_width_rows`: the first
changes the treatment of *valid* adjacent measurements, and the latter two
only change what width is sufficient. Avoid weakening those settings just
to compensate for missing data.

`TopEntryBridgeDetector(gap_radius_px=...)` already provides a **geometric**
`candidate_bridge`, not proof of a bridge or even of sensor dropout.
`BridgeDrapeDetector(max_gap_px=...)` can jump small *observed* gaps in depth-
difference mode, but it cannot traverse invalid or excluded pixels. It does
not supply the missing-depth hypothesis below. Hanging shape measurements
also remain separate: none of the inferred pixels are inputs to measured
width, sag, drape, or depth.

## Opt in with the existing mask and depth

```python
from depth_image_lens import ObjectOnlyGraspDetector, TopEntryBridgeDetector

grasp = ObjectOnlyGraspDetector(invalid_gap_max_px=3).detect(
    depth_m, cloth_mask, depth_unit="m", difference=background_minus_current_m,
    diagnostics=True,
)
top = TopEntryBridgeDetector(invalid_gap_max_px=3).detect_mask(
    depth_m, cloth_and_grippers_mask, depth_unit="m",
    diagnostics=True,
)

print(grasp.held, grasp.bilateral, grasp.metrics["visible_width_px"])
print(grasp.metrics["invalid_depth_gap"])
print(top.bridge, top.metrics["invalid_depth_gap"])
```

The ordinary masks can already be cut by missing depth. No RGB or separate
cloth segmentation is required. If an aligned bool HxW `gap_support_mask` is
available, pass it to restrict allowed missing pixels and their immediate
measured endpoints. It is an optional filter, not additional depth or proof
of object identity. Do not pass a depth-intersected mask as that filter if it
excludes the missing stripe. The default `invalid_gap_max_px=0` does not compute
an additional result.

Each missing run must have at least three contiguous eligible measured pixels
on each side. The flank depth must be locally coherent separately on each side,
along the run and between adjacent aligned scanlines. The two depth levels do
not have to agree: a fold or wrinkle can present a real step. The algorithm
connects a geometry hypothesis only. It does not interpolate, output, or add
any depth values. Registered bursts of an unchanged scene may offer actual
additional observations for intermittent dropout; persistent projector shadow
or unregistered motion can defeat that strategy.

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `invalid_gap_max_px` | 0 | Largest number of consecutively missing pixels allowed along an individual horizontal or vertical traversal, 0–8. Start at 1–3 and inspect the inferred mask. |
| `invalid_gap_flank_depth_jump_m` | 0.02 | Maximum local adjacent camera-Z change within either flank, including across neighboring support lines, in metres even with mm input. The ordinary depth jump limit also applies. Start here; raise only if actual same-side measurements are coherent but exceeded the limit. |
| `invalid_gap_depth_jump_m` | None | Optional explicit across-gap depth difference cap retained for callers who want it. Leave unset to allow genuine fold steps. |
| `invalid_gap_min_lines` | 3 | Adjacent parallel lines supporting aligned or touching gap spans. A top bridge also requires at least its configured path width in support lines. |

Only nonpositive/nonfinite current depth is repairable. If the sensor uses a
different invalid sentinel, normalize that value to zero or NaN before
calling. Valid desk/background depth, valid pixels that were not selected,
and caller-excluded or uncertain areas are not repairable. These exclusions
also apply to `observed_mask=False` and, for object-only background comparison,
`background_valid=False`. An omitted `observed_mask` leaves the image domain
eligible for consideration; invalid depths are still never observed evidence.
Do not mark intentional exclusions or cropped/unobserved regions as usable to
force a candidate. For object-only background comparison, any supplied
`background_valid` should describe the **background** validity, not a combined
current/background mask; otherwise current dropout is deliberately excluded.

Each candidate connects just two measured components, not chains of pieces.
Object-only candidates need two eligible elevated fragments and must still
meet `min_width_px` and `min_width_rows`. The candidate row width can include
inferred pixels, but inferred pixels may occupy at most a quarter of the
two-fragment-plus-gap area. The top-entry candidate needs intact measured
attachments to both original entries, a thick candidate route and a crossing
below the entry band and minimum drop. It also needs measured foreground between
the top-entry projections. For a horizontal crossing, both measured flanks
must extend into that strip; simply seeing two straight grippers on either side
of missing depth is insufficient. It neither invents entries nor repairs a
stripe crossing both legs of a U when three measured fragments are required.

The parallel runs can track a stepped, wandering, or diagonal narrow band,
even when it changes width. The implementation can use a locally supported
subset of the band; it never fills over-width parts. Adjacent matched endpoints
may shift at most 3 px across scanlines and the runs must touch or overlap in
their projections. Sharp zigzags with no
compatible consecutive runs, and a thin residual observed link already joining
both sides into one component, can remain unresolved. Do not widen the
threshold aggressively to force those cases.

## Read the result

- The original measured fields and `candidate_bridge` are unchanged.
- `metrics["invalid_depth_gap"]["candidate"]` is a supplementary hypothesis.
  For object-only width, `candidate_sustained_width_px` includes the inferred
  crossing, unlike `visible_width_px`. For top-entry, no promotion to
  `bridge="yes"` occurs.
- With diagnostics, a positive **candidate** includes
  `invalid_depth_gap_inferred_mask` (only the newly hypothesized pixels) and
  `invalid_depth_gap_candidate_mask` (hypothesis including measured pieces).
  Top-entry also provides `invalid_depth_gap_candidate_core_mask`.
- The candidate reports `supporting_lines`, `max_missing_run_px` and
  `median_cross_gap_depth_step_m` from the matched flanks. The median cross-step
  is descriptive, not a default pass/fail threshold.
- Two separate aligned pieces with a genuinely unmeasured physical gap can
  produce exactly the same sensor input as a folded cloth with dropout. Even
  locally coherent measured flanks cannot distinguish these cases. Never treat
  this result as an observed grasp or completed bridge; use additional
  observation from a changed viewpoint or independent evidence if needed.

Run `python examples/example_oblique_invalid_depth_gap.py` to compare the
measured and candidate paths on an artificial stripe. The synthetic checks are
not a measured camera-accuracy guarantee.
