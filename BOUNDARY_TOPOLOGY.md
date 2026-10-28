# Optional boundary continuity and topology

`OverheadBoundaryDetector(trace_boundaries=True)` adds `result.topology` without
replacing the original measured color/depth boundary masks, lines, islands or
metrics. The default is off, so existing callers retain their detection behavior.
This is a graph of image evidence, not a semantic wrinkle map or physical boundary
identity tracker. The `BOUNDARY.md` input/registration/missingness contracts apply.

```python
from depth_image_lens import OverheadBoundaryDetector
model = OverheadBoundaryDetector(
    trace_boundaries=True, color_weak_ratio=0.5, depth_weak_ratio=0.5,
    max_gap_px=6, gap_angle_deg=25, tangent_window_px=5,
    corner_turn_deg=45, corner_window_px=5,
)
result = model.detect(color_rgb, depth, color_order="RGB",
                      registered_to_depth=True, depth_scale_m=0.001,
                      object_mask=visible_object_mask, diagnostics=True)
for kind in ("color", "depth"):
    graph = result.topology[kind]
    for segment in graph["segments"]:
        xy = segment["points_xy"]  # ordered [x,y] processing pixels
        evidence = segment["evidence"]  # same length: strong, weak, inferred
        print(kind, segment["id"], segment["start_node"], segment["end_node"])
result.save_debug("topology.png", color_rgb, topology_labels=True)
```

A supplied mask is optional: the existing table-color, background-reference and
rectangle methods can provide the domain. Poor segmentation still compromises the
graph. Missing optional depth leaves the color graph available. Empty explicit masks
remain empty. Unscoped depth fallback is labelled `full_frame_without_object_identity`
and may include the table/object exterior; do not call it an internal object graph.

## What is computed

1. **Seeded weak continuation.** Existing strong boundary pixels seed 8-connected
   components of a weaker evidence mask. Isolated weak components are not retained.
   For color, darkness OR gradient is compared to its original threshold multiplied
   by `color_weak_ratio`. For depth, adjacent residual steps use
   `max(depth_step_m * depth_weak_ratio, 2 * depth_noise_m)`; the noise floor remains.
   This is connectivity-based hysteresis, not Canny nonmaximum suppression. Texture,
   prints or shadows can connect into a large component when the weak threshold is low.
2. **Centerline representation.** Deterministic two-pass Zhang-Suen thinning produces
   a pixel skeleton of the retained evidence band. Neighbor links use 8-neighborhood
   connectivity but omit diagonal shortcuts when an orthogonal route exists, avoiding
   artificial triangles at raster right angles. Thin curved paths and holes/cycles
   can survive. A wide dark stripe with only its two gradient sides observed can
   produce two paths, not one inferred physical centerline. No subpixel fit is claimed.
3. **Optional short reconnection.** Skeleton endpoints must have observed directional
   support, a mutually unique compatible partner, compatible outward tangents, and
   lie in the same allowed domain component. The raster connector must remain in
   observed domain (for depth, valid fitted depth) and avoid crossing/side-touching
   existing paths or another proposed connector. Ambiguous partners are left separate.
   No global dilation joins nearby parallel lines. Depth reconnection uses valid
   spatial support and tangent compatibility, not an inferred same-layer or identical
   signed-height-step identity. A gap is a hypothesis, never a
   manufactured darkness/depth measurement. Color-only interpolation need not have
   depth support. A small true corner hidden inside a gap may remain disconnected
   because the tangent criterion favors approximately collinear continuations.
4. **Graph and corner splitting.** Adjacent high-degree pixels form a junction region.
   Paths connect endpoint/junction regions; junction arm correspondence is left
   unassigned. On each degree-two path, direction change is measured using chords
   approximately `corner_window_px` arclength on either side. Local nonmaximum
   suppression over the same distance avoids repeated detections around one bend.
   Turns at least `corner_turn_deg` split the path at a corner node. Zero degrees is
   straight, 90 is a right-angle turn, 180 is a reversal. A corner is degree two;
   a junction is a separate connectivity event and survives corner threshold changes.
   A smooth curve can split if its turn over the chosen window exceeds the threshold.
   Endpoint neighborhoods with insufficient support do not receive a corner estimate.

No graph connects color and depth paths into a single identity: the existing
`agreement_mask` remains a proximity measurement only. Weak/interpolated parts are
not added to the original agreement or other measured masks. There is no temporal
tracking or learned classification. A crossing does not establish which sheet is on top.

## Parameters and tuning

| Parameter (default) | Meaning |
|---|---|
| `trace_boundaries=False` | Opt into graph extraction |
| `color_weak_ratio=1`, `depth_weak_ratio=1` | (0,1]; lower values retain more weak connected evidence; strong thresholds unchanged |
| `max_gap_px=0` | Integer 0..20, maximum missing centerline samples; 0 disables reconnection; endpoint distance must also be <= this value +1 |
| `gap_angle_deg=25` | 0..60 degrees, maximum error of **both** outward endpoint directions relative to the connector |
| `tangent_window_px=5` | Integer 2..50, path arclength support for gap directions and reported junction branch angles |
| `corner_turn_deg=45` | (0,180] degrees; lower values split smaller bends; values >90 can retain a right angle within one polyline |
| `corner_window_px=5` | Integer 2..50, arclength scale for turn estimates and corner suppression; independent of gap tangent support |
| `min_path_length_px=3` | Nonnegative Euclidean arclength; shorter segments remain with `status=short_path`, not silently removed |

These are provisional processing-pixel parameters, not sensor-calibrated physical
sizes. Resizing changes the needed pixel lengths. Skeleton endpoint retreat means
a 3-column gap in a thick painted band can require a larger centerline gap setting.
Inspect the reported `missing_pixels` rather than assuming equality to a painted gap.
The example values above demonstrate sensitivity, not a universal recommended preset.
Start at weak ratio1 and gap0; inspect strong evidence, then lower the weak ratio
modestly. In textured scenes, ratio0.5 can dramatically increase spurious branches.
Enable gaps last, keep direction tolerance small and inspect orange inferred points.
Tune corner window for the image resolution before choosing turn threshold. Do not
reduce fragmentation by accepting arbitrary parallel/crossing connections.

Work is bounded: over250,000 retained pixels or over50,000 skeleton pixels produces
an explicit topology `insufficient` status/reason, while original detection outputs
remain available. Resize or restrict the domain instead of treating empty graph
fields as absence of a boundary. These limits are computational guards, not confidence.

## Output schema

`result.topology` is empty when disabled. When enabled it contains `color` and `depth`:

- `status`, optional `reason`, `domain`, `counts`, `warnings`, `gap_rejections`.
- `nodes`: dictionaries with local `id`, `kind` (`endpoint`, `junction`, `corner`,
  `isolated`), representative `xy`, `pixels_xy` (junction region), `component` and
  incident `segment_ids`. Corner nodes include `turn_deg`; junctions include pairwise
  `branch_angles_deg` referring to segment IDs. Node IDs are not stable across settings.
- `segments`: local `id`, graph `component`, `start_node` / `end_node`, `closed`,
  ordered `points_xy`, same-length `evidence` strings, `length_px`, `status` and
  `measurements`. Segment endpoints attach to node-region pixels, not necessarily
  the representative node coordinate. Shared endpoints can appear in multiple paths.
  Pure loops have `closed=True`, repeated first/last point, and null endpoint node IDs;
  if corners are selected, a loop becomes connected corner-to-corner segments.
- `measurements`: mean/p95 of darkness/gradient or adjacent `step_strength_m` at
  non-interpolated centerline samples. Inferred samples do not receive fabricated
  measurements. These are not integrated physical curvature or boundary-height truth.
- `connections`: ID, endpoint coordinates, inferred pixel coordinates, missing sample
  count, endpoint distance, directional errors and explicit inferred status.

Coordinates are integer **[x,y]** in the processed image; arrays use **[y,x]**.
Lengths are Euclidean pixel arclength. Junction branch angles are between outward
arms (0..180), distinct from the corner turn convention. Connected-component IDs
refer to raster skeleton connectivity, not different physical wrinkles.

With diagnostics enabled, `diagnostics['color_topology']` and `['depth_topology']`
contain bool HxW `retained_mask`, `observed_centerline_mask`, and `inferred_mask`.
The retained band includes weak threshold evidence. Graphs remain available with
`diagnostics=False`. `to_dict()` contains graphs but excludes these arrays; use
NumPy NPZ separately. Original `lines[*].band_contours_xy` remain band outer contours
and are not renamed into centerlines.

Debug output adds separate color/depth graph views: green strong, blue weak, orange
inferred, magenta corners, yellow junctions, pale endpoints. Optional
`topology_labels=True` displays node IDs; dense graphs can be cluttered. It preserves
RGB/BGR output conventions and explicit non-overwriting save behavior.
`examples/example_boundary_topology.py` is a minimal analytic example. Dataset-specific
comparison scripts and real image galleries remain outside Git.

## Method references and limits

The connectivity principle of high/low thresholds is described in the official
[OpenCV Canny tutorial](https://docs.opencv.org/4.13.0/da/d22/tutorial_py_canny.html).
This feature retains the existing darkness/gradient bands rather than replacing them
with Canny. Zhang-Suen thinning is a standard binary thinning method also documented
by [OpenCV](https://docs.opencv.org/4.13.0/df/d2d/group__ximgproc.html); this small
NumPy implementation introduces no contrib dependency or copied external code.
Topology preserves image evidence, including undesired prints/seams/shadows. It does
not establish physical wrinkle/overlap correctness. Existing real datasets have no
internal line/junction annotations, so real comparisons measure sensitivity and
operational consistency, not accuracy.

A path that returns to the same junction has two incident arms. Junction `arms`
records `segment_id` and `end` (`start`/`end`); `branch_angles_deg[].arms` indexes
this list. Thus two distinct arms can refer to the same segment ID.


## Optional junction grouping hypotheses

A thick raster X can thin into two degree-three junctions connected by a short
observed path, rather than one degree-four node. Existing corner/gap settings do
not consolidate junctions. This is a localization/representation ambiguity, not
missing connectivity. The raw graph must remain available.

`junction_cluster_px=0` (default) disables optional grouping. Set a small positive
integer (0..20 processing pixels) to request `graph['junction_grouping']`. This
never merges or deletes `nodes`, `segments`, raw counts, masks or measurements.
It adds disjoint two-node grouping **hypotheses**, with `grouped_junction_count`
for consumers choosing that coarser representation. Example: a reviewed thick X
has two raw junctions 4px apart; value4 yields one group with four external arms.
Value3 leaves them separate. This is not a universal recommended tolerance.

Only two degree-three junctions linked by one short, entirely observed segment
are eligible. Their four distinct external arms must have enough observed,
approximately straight support outside the junction neighborhood. Fits use the
path interval from `max(junction_cluster_px, tangent_window_px)` to that distance
plus twice `tangent_window_px`, requiring at least four samples and one window of
arclength. Unsupported, curved, interpolated, looping/shared arms are not grouped.
The four lines must admit exactly one opposing pairing and a common local
intersection. Competing candidate groups sharing a node are all rejected.

| Parameter (default) | Effect |
|---|---|
| `junction_cluster_px=0` | Integer0..20; maximum observed connector arclength; larger values consider more distant junction pairs and move fit support outward |
| `junction_cluster_residual_px=1` | (0,3] pixels; maximum arm-line and common-intersection perpendicular residual; also bounds intersection extension outside the connector bounding box |
| `junction_cluster_angle_deg=15` | [0,45] degrees; maximum deviation from180 degrees for each opposing pair; larger is less restrictive |

The regression set keeps same-side and opposing nearby T nodes separate at the
provisional residual1px/angle15deg settings. These checks cannot prove one physical
crossing: genuinely distinct junctions with indistinguishable image geometry may
still satisfy them. Raising residual tolerance can group offset distinct T nodes.
Leave grouping disabled unless the chosen resolution and real examples justify it.
Short external arms, curves, missing support or complex junction clusters may
remain ungrouped. No degree-five or multi-node general clustering is implemented.

`junction_grouping` contains `status`, `groups`, `grouped_junction_count` and
`rejections` (candidate-pair counts, not physical error counts). Each group records
raw `nodes` IDs, `internal_segments`, `external_arms` (segment ID and start/end),
float `center_xy`, connector arclength, fitted-line/intersection residuals in
pixels, opposing-arm index pairs and directional errors in degrees. Its status
is `inferred_junction_group_not_physical_identity`. No new image pixels are created.
The debug image marks the inferred center with a cyan ring while retaining yellow
raw junctions. `junction_grouping` is absent when the topology computation itself
is insufficient. `counts['junctions']` always refers to raw junctions.
