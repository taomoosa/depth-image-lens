# Optional boundary continuity and topology

`OverheadBoundaryDetector(trace_boundaries=True)` adds `result.topology` without
replacing the original measured color/depth boundary masks, lines, islands or
metrics. The default is off, so existing callers retain their detection behavior.
This is a graph of image evidence, not a semantic wrinkle map or physical boundary
identity tracker. The [boundary input, registration and missingness contracts](BOUNDARY.md) apply.

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
Start at weak ratio 1 and gap 0; inspect strong evidence, then lower the weak ratio
modestly. In textured scenes, ratio 0.5 can dramatically increase spurious branches.
Enable gaps last, keep direction tolerance small and inspect orange inferred points.
Tune corner window for the image resolution before choosing turn threshold. Do not
reduce fragmentation by accepting arbitrary parallel/crossing connections.

Work is bounded: over 250,000 retained pixels or over 50,000 skeleton pixels produces
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

## Component sizes and maximum trails

After detection with `trace_boundaries=True`, call the result method:

```python
report = result.measure_components('color')  # Or 'depth'; separate ID spaces.
for component in report['components']:
    print(component['id'], component['span_xy_px'], component['bbox_size_px'])
    print(component['path_kind'], component['status'], component['max_trail_length_px'])
    if not component['exact']:
        print('Bounds:', component['lower_bound_length_px'], component['upper_bound_length_px'])
    for part in component['trail']['parts']:
        print(part['segment_id'], part['reversed'], part['points_xy'])
```

This is an on-demand measurement of the **existing compressed segment graph**.
It does not rerun segmentation, join gaps, change graph IDs, or mutate any existing
result. `diagnostics=True` is not required. `kind` accepts only `'color'` or
`'depth'`. If tracing is disabled or that modality's topology is insufficient,
the report has `status='insufficient'`, a `reason`, and an empty components list.
Otherwise `status='computed'` means measurements were attempted; inspect each
component's `status` and `exact`, since cyclic searches can be bounded.

**Size:** `bbox_xyxy=[xmin,ymin,xmax,ymax]` uses inclusive pixel-center extrema,
including both segment points and node-region `pixels_xy` (also isolated points).
`span_xy_px=[xmax-xmin,ymax-ymin]` is the coordinate extent.
`bbox_size_px=[xmax-xmin+1,ymax-ymin+1]` is the integer bounding-box width/height.
An isolated pixel therefore has span `[0,0]` and bbox size `[1,1]`.
These describe the centerline graph, not the full thickness of the original
evidence band. `result.lines[*].pixels` is band area, and its IDs are not topology
component IDs. `touches_image_boundary` only tests actual graph pixel contact;
False does not prove absence of clipping or missing observations.

**Trail definition:** one segment may be traversed at most once, in either
direction. A junction vertex may be revisited. This is an *edge-simple trail*,
not a vertex-simple path, shortest-path diameter, or unrestricted sum of branches.
Corners are ordinary degree-two vertices: a trail continues through corner splits.
Self-loops and parallel segments remain distinct edges. Only nodes with
`kind='endpoint'` are terminal endpoints.

| Component | Measurement convention |
| --- | --- |
| At least two terminal endpoints | Longest trail between **distinct** terminal endpoints; `path_kind='endpoint_trail'`. It may go around a loop before continuing to its other endpoint, using each segment once. |
| Zero or one terminal endpoint | Longest nonempty closed trail, starting and ending at the same graph vertex; `path_kind='closed_trail'`. `endpoint_pair_available=False` and `max_endpoint_trail_length_px=None` distinguish it from an endpoint measurement. |
| Pure cycle | One finite circuit, whether stored as one node-free loop or multiple corner-to-corner segments. |
| Loop with one tail | Closed-trail convention measures the loop. The tail cannot be included by going out and back along the same edge. |
| No traversable segments | `status='no_edges'`, exact length 0, empty trail parts; bbox still includes node pixels. |
| No endpoint pair and no cycle | `status='no_closed_trail'`, exact determination that no nonempty closed trail exists; lengths are None. |

A Y/T component's maximum endpoint trail omits the unused branch. Two loops joined
by a bridge cannot both belong to one closed trail, because returning across that
bridge would reuse it. A closed trail can include multiple loops meeting at the
same junction when no edge is repeated. Short segments marked `short_path` remain
part of this measurement; `min_path_length_px` does not remove them.

**Length scope:** `length_px` values are summed along the chosen oriented segments.
The existing graph collapses a junction region into a vertex and omits its internal
pixel links. Travel *inside* that region is unmeasured and contributes zero here.
Thus an exact result is exact **for that compressed graph**, not an exact full
pixel-skeleton distance or a physical material length. Even with depth/intrinsics,
these results are in processing-image pixels, not metres.

`trail.parts` contains ordered, detached dictionaries with `segment_id`, `reversed`,
`start_node`, `end_node`, `points_xy`, `evidence`, and `length_px`. Coordinates and
evidence are reversed together when traversed backward. Do not concatenate the
parts and blindly add distances across different junction attachment pixels:
that invents a connector length. Such gaps are explicitly recorded in
`trail.junction_transitions` with `status='junction_travel_unmeasured'` and zero
included length. Closed-trail closure can also have such an unmeasured junction
transition. A node-free pure loop has null node IDs and repeated first/last point.
The report also supplies `start_xy`/`end_xy` inside `trail` for visualization.

Each component includes `endpoint_node_ids`, `junction_node_ids`, `node_ids`,
`segment_ids`, `total_segment_length_px` (all edges, deliberately separate from
the maximum), `contains_inferred`, and `contains_weak`. The selected trail has
`uses_inferred` and `uses_weak`. If `max_gap_px>0` was enabled in detection,
inferred connectors contribute to both graph length and connectivity; they are
not new measured boundary evidence. Leave `max_gap_px=0` to avoid such connectors.
Endpoint nodes can arise from thresholds, ROI/cropping or depth missingness, and
must not automatically be interpreted as physical ends of a wrinkle.

### Exactness and bounded computation

```python
report = result.measure_components(
    'depth', max_search_states=100_000, max_search_edges=24,
)
```

- `max_search_states`: Python int, default 100000, range 0..1000000. This is a
  **shared** budget of distinct exhaustive-search states across all components in
  one call; deterministic component-ID order consumes it. Raising it can establish
  more exact results at higher CPU/memory cost. Zero disables this search.
- `max_search_edges`: Python int, default 24, range 0..64. General cyclic search is
  skipped above this number of compressed segments per component. Corner splits
  count too. Raising it may admit much harder searches, still bounded by states.
- Trees use an exact weighted terminal-diameter algorithm. Connected graphs that
  admit a full Euler trail/circuit under the selected endpoint convention use an
  exact linear-time traversal. These fast cases work even with both search limits
  zero. General cyclic branching graphs are **not** claimed to have a polynomial
  maximum-trail algorithm.

An exact solution has `status='exact'`, `exact=True`, and numeric
`max_trail_length_px`. The appropriate specialized field is also filled:
`max_endpoint_trail_length_px` or `max_closed_trail_length_px`.
An unfinished or skipped general search has `status='bounded'`, `exact=False`,
`max_trail_length_px=None` and both specialized maxima None. Its representative
trail establishes `lower_bound_length_px`; `upper_bound_length_px` is the sum of
all component segment lengths, which can be loose. **The representative lower
bound must not be displayed as the maximum.** `reason` distinguishes edge/state
limits, and `search_states` records work. The report summary counts bounded
components and total search states. Ties return one deterministic representative,
not every maximizing trail. For no-edge components the generic exact length is
zero; a nonempty closed-trail maximum is still None.

The returned dictionary is JSON-compatible and does not share mutable coordinate
lists with the original topology. There is no cached automatic detection overhead.
For executable examples with default/exhaustive limits and pixel overlays, run:

```bash
PYTHONPATH=src python examples/example_boundary_components.py
PYTHONPATH=src python examples/example_boundary_components.py --output-dir /existing/parent/new-directory
PYTHONPATH=src python examples/example_boundary_components.py --max-search-states 0 --output-dir /existing/parent/bounded-run
```

The example writes the topology debug view, separate representative trails and
bounding boxes for color/depth, and measurement JSON. It draws each path part
independently so unmeasured junction travel is not shown as an observed connector.
These generated scenes test API geometry, not real-image boundary accuracy.

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
`examples/example_boundary_topology.py` is a minimal analytic example.

## Method references and limits

The connectivity principle of high/low thresholds is described in the official
[OpenCV Canny tutorial](https://docs.opencv.org/4.13.0/da/d22/tutorial_py_canny.html).
This feature retains the existing darkness/gradient bands rather than replacing them
with Canny. Zhang-Suen thinning is a standard binary thinning method also documented
by [OpenCV](https://docs.opencv.org/4.13.0/df/d2d/group__ximgproc.html); this small
NumPy implementation introduces no contrib dependency or copied external code.
Topology preserves image evidence, including undesired prints/seams/shadows. It does
not establish physical wrinkle/overlap correctness. Measuring accuracy requires
independent internal line/junction annotations; see
[validation limits](BOUNDARY.md#method-scope).

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
for consumers choosing that coarser representation. For example, an eligible pair linked by a 4 px observed connector can group at
value 4 but not value 3. This is not a universal recommended tolerance.

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
| `junction_cluster_px=0` | Integer 0..20; maximum observed connector arclength; larger values consider more distant junction pairs and move fit support outward |
| `junction_cluster_residual_px=1` | (0,3] pixels; maximum arm-line and common-intersection perpendicular residual; also bounds intersection extension outside the connector bounding box |
| `junction_cluster_angle_deg=15` | [0,45] degrees; maximum deviation from 180 degrees for each opposing pair; larger is less restrictive |

These geometric checks cannot prove one physical crossing: genuinely distinct junctions with indistinguishable image geometry may
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

## Outer proximity selection

Use `result.filter_outer_proximity(near_px=8., reject_fraction=.7)` to obtain
kept/excluded/pending selections without modifying the original graph. See
[distance, holes, clipping and debug API](BOUNDARY_PROXIMITY.md).
