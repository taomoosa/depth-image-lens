# Filter topology candidates near the object exterior

`BoundaryResult.filter_outer_proximity` is optional postprocessing. It returns a
`BoundaryProximityResult` with `kept`, `excluded`, `pending` and `ignored` record
lists, without changing the source masks, graph nodes/segments, IDs or measurements.
It selects image evidence; a real fold near the silhouette can be excluded by this
policy. Retaining a path does not prove a physical fold or overlap.

```python
from depth_image_lens import OverheadBoundaryDetector

result = OverheadBoundaryDetector(trace_boundaries=True).detect(
    color_rgb, color_order='RGB', object_mask=object_mask)
selection = result.filter_outer_proximity(
    near_px=8., reject_fraction=.7, include_holes=False)
for record in selection.kept:
    segment = record['segment']
    print(record['channel'], record['segment_id'], segment['points_xy'],
          record['near_fraction'], record['min_distance_px'])
selection.save_debug('proximity.png', color_rgb, overwrite=False)
```

Use existing foreground/background segmentation instead of `object_mask` when
appropriate; this filter always reads **result.object_mask**. No depth or diagnostics
are needed by the filter. To obtain depth paths, provide registered depth to detect
as usual. Both color and depth topology channels are processed independently. No
additional OpenCV dependency beyond the existing boundary/debug extra is introduced.

## Settings and exact distance contract

```python
result.filter_outer_proximity(
    *, near_px=8., reject_fraction=.7, include_holes=False,
    observed_mask=None)
```

| Setting | Default/type/unit | Effect |
|---|---|---|
| `near_px` | 8.0, finite nonnegative number, processing px | Points at distance **<=** this value count as near. Larger values expand the exclusion band and the artificial-border uncertainty band. Boolean values are rejected. |
| `reject_fraction` | 0.7, finite number in **(0,1]** | Exclude when near observed point count / observed point count is **>=** this value. Lower values exclude more paths. Zero/boolean values are rejected. |
| `include_holes` | False, bool | False ignores background holes when computing proximity; True treats holes as additional boundaries. |
| `observed_mask` | None, optional bool HxW | Known observed ROI, same grid. None assumes the entire image is observed. Always preserve/provide artificial crop provenance. |

Distance is Euclidean distance from an object pixel center to the nearest zero
pixel center of its **own component's** mask. With include_holes=False, holes are
filled separately for each 8-connected component before computing distances. A
separate foreground island inside another component's hole therefore retains its
own exterior. No global filling merges identities. With include_holes=True, use the
original per-component mask. The distance transform is explicitly zero-padded;
image edges never accidentally have infinite distance. An axis-aligned boundary
object pixel is normally distance1px, not0: this is **not** exact distance to contour
pixel centers or a subpixel physical silhouette. `near_px=0` therefore normally has
no near object pixels.

The denominator is **unique XY coordinates with strong or weak evidence** in the
source segment. Repeated samples/closed endpoints count once. Inferred points are
excluded from numerator and denominator. If the same coordinate occurs as observed
and inferred, its observed occurrence counts once. `inferred_unique_count` counts
coordinates carrying inferred provenance, so it may overlap the observed count.
The fraction is point-count based, not arclength weighted; diagonal sampling and
resolution affect it. Choose pixel thresholds for the actual processing grid.

A large internal line that only touches the exterior at its endpoints is normally
kept at a high fraction threshold. A minimum-distance-only test would incorrectly
remove it. The returned minimum is diagnostic; it does not drive selection.

## Continuation, clipping and graph identity

- `kept`: candidate with valid domain/evidence and near fraction below the limit.
- `excluded`: candidate with valid domain/evidence and fraction at/above the limit.
- `pending`: missing/invalid evidence, noninteger/nonfinite/out-of-range points,
  path outside the object/observed domain, multiple object components, uncomputed
  topology, no observed points, or proximity to a known artificial border.
- `ignored`: original status is not `candidate` (including `short_path`). The full
  source snapshot remains available; this filter does not promote short paths.

Object contact with the image edge or an explicitly unobserved ROI is artificial
border evidence. If **any observed sample** is within near_px of such contact,
that segment is conservatively pending, even if other samples are far inside.
Other independent segments continue normally. This special uncertainty rule differs
from the fraction test on an ordinary observed exterior. It can withhold a long path
with a clipped endpoint; inspect the reason instead of treating it as excluded.
An undisclosed internal crop in a supplied object mask cannot be recognized. A full
image mask does not confirm a physical silhouette; its border contacts are recorded,
while sufficiently interior paths can still be kept as local image evidence.

With no topology, channels are `not_available` and there are no records. Insufficient
channel status is retained. Empty graphs return summary status `no_segments`, which
does not prove that physical internal boundaries are absent. Empty object masks with
existing paths cannot produce accepted geometry. Invalid global settings/mask shape
raise clear ValueError; local bad segments do not stop unrelated candidates.

The returned records are **selections of original segments**, not a newly coherent
graph. Original nodes may still reference excluded segments. Use `(channel,
segment_id)` to map records back to `result.topology[channel]`. Do not substitute a
filtered segment list into the original graph and claim that its connectivity has
been rebuilt. No physical or temporal identity is assigned.

## Returned fields and debug output

`BoundaryProximityResult` contains `records`, `parameters`, `summary`, copied masks
`object_mask`, `observed_mask`, `near_mask`, `uncertain_border_mask`, and float32
`distance_px` (NaN outside the visible object). `kept/excluded/pending/ignored` are
convenience record lists. The original source result is not mutated; each record's
`segment` is an independent deep copy preserving ordered points/evidence/measurements.

Each record includes channel/segment_id/source_status/status, reasons,
observed_points_xy, observed_count, inferred_unique_count, invalid_input_point_count,
near_observed_count, distances_px, min_distance_px, near_fraction,
uncertain_border_count and object_component_ids when available. `distances_px` aligns
with the lexicographically sorted unique observed_points_xy, not the original path
order. Unavailable measurements are null/empty, not zero certainty. Invalid input
point counts count original occurrences, not unique points.

`to_dict()` excludes image arrays and converts any nonfinite values retained in a
malformed source snapshot to null for strict JSON. The in-memory source copy remains
unchanged. Summary gives channel states, class counts, artificial-border pixel count
and whether observation-domain provenance was supplied. Settings also record the
distance/sample/frame policies. To replay thresholds from JSON, pass only near_px,
reject_fraction and include_holes; observed_mask is a separate array input.

```python
image = selection.render_debug(color_rgb, color_order='RGB', show_ids=True)
path = selection.save_debug('proximity.png', color_rgb, color_order='RGB',
                            overwrite=False, show_ids=True)
```

Renderer signature: `render_debug(color=None, *, color_order='RGB', depth=None,
depth_scale_m=.001, show_ids=False, panel=True)`. It returns uint8 in the requested
RGB/BGR order, with original paths on the left and green kept/red excluded/yellow
pending/gray ignored paths on the right. A red tint shows the proximity band;
magenta shows known artificial-border uncertainty. Use depth percentile display or
mask shading when color is absent. It is a comparison display, not a distance heatmap.
Saving follows the existing explicit PNG, non-overwrite, existing-parent contract
and returns a Path. No files are written during detection/filtering/rendering.

Run `python examples/example_boundary_proximity.py --output NEW_IMAGE.png`.
Adjust `--near-px`, `--reject-fraction`, `--include-holes` and use a new output path.
The example constructs analytic color bands, runs the actual detector, then filters
its graph. Evaluation fixtures and real-data comparisons remain outside Git; no
real physical boundary accuracy is established by this proximity heuristic.
