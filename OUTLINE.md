# Ordered external contours and exposed-length candidates

`OuterContourDetector` decomposes the visible external silhouette of each selected
component into a cyclic sequence of straight intervals, curved intervals and corner
events. It is an image-geometry helper, **not** material-edge identification or
proof that an original deformable-object side is exposed. An occlusion or fold can
produce the same silhouette. Hole boundaries are returned separately and never
enter external-edge queries. No inner wrinkle graph is reused as an exterior.

Install the optional OpenCV dependency with `pip install -e '.[outline]'` (the
existing boundary/corner/debug extras also supply it). No new model or service.

```python
from depth_image_lens import OuterContourDetector

# Strict bool HxW visible object region; e.g. boundary_result.object_mask.
# Inspect segmentation quality first; use mask_uint8 != 0 for a binary PNG mask.
model = OuterContourDetector(corner_turn_deg=45, line_error_px=1.5)
result = model.detect(object_mask, observed_mask=observed_region, diagnostics=True)
for contour in result.contours:
    print(contour['id'], contour['sequence'])
    for section in contour['segments']:
        print(section['kind'], section['points_xy'], section['fit_length_px'])
query = result.query_edge_length(100, unit='px', tolerance=5, min_support_px=20)
image_rgb = result.render_debug(color_rgb, color_order='RGB')
path = result.save_debug('outline.png', color_rgb, color_order='RGB')
```

Color is only a debug background and is not required by detection. A segmentation
result can be composed with this class; no perfect mask or semantic segmentation
accuracy is implied. Detection does not modify inputs or write files. Empty or tiny
regions produce explicit empty/insufficient results, not invented large contours.

## Method and edge ownership

1. OpenCV `findContours(..., RETR_CCOMP, CHAIN_APPROX_NONE)` extracts the visible
   bool-mask boundary without polygon simplification or convex hull. Nested foreground
   islands are separate outer components; their enclosing background holes remain holes.
   Raw pixel-center points are canonicalized clockwise in image coordinates, starting
   at the topmost, then leftmost point. Holes use the same canonical convention but
   have an explicit hole role. Zero-area chains are labeled degenerate.
2. The closed contour is uniformly sampled at approximately1px arclength. Periodic
   Gaussian coordinate smoothing is used **only** for tangent/turn measurements.
   Chords on each side of a point measure signed turn at the selected support scale.
   Thresholded turns undergo cyclic distance suppression; the array seam is not an
   endpoint. Positive turns are convex, negative concave in the chosen orientation.
3. Corner events and changes in boundary observability divide the raw cyclic edges.
   A supported whole interval that fits one line can remain straight. Other intervals
   are partitioned by local turn. Brief low-turn patches need `min_straight_length_px`
   support; they otherwise remain part of a curve. Adjacent same-kind sections merge
   without crossing a corner or observation cut. Whole-section TLS line error and
   arc/chord ratio prevent a long shallow curve from being accepted solely because
   every local turn is small. Short unsupported sections are `unknown`, not discarded.
4. Each section refers to consecutive **raw edges** and retains raw points and line-fit
   measurements. No contour gap is filled; fit endpoints are separate derived geometry.
   A smooth full loop can be one curve with no corners. Tangent straight/curve joins
   are section boundaries without corner events. Increasing corner threshold may turn
   a formerly sharp event into a curved connecting section; junctions are not imposed
   on a simple cyclic contour. A one-pixel self-touch/repeated point is flagged ambiguous.

Raw contour points are `points_xy` with no repeated closing endpoint. Edge i runs
from point i to `(i+1) % edge_count`. A segment owns `edge_count` edges starting at
`start_index`; `end_index` is the final endpoint modulo contour size. Its points
include both endpoints; adjacent sections share an endpoint but **never an edge**.
Summed section arc lengths equal the closed raw perimeter. `sequence_start_index`
may differ from raw point0 because a section can cross the canonical seam. Follow
`sequence`, not a length-sorted list, to traverse one complete cycle.

Pixel-center coordinates differ from outer cell-edge dimensions (a mask spanning
x=40..180 has a140px center-to-center side). Raster diagonal staircases inflate
raw arc length compared with Euclidean physical edges. Resizing changes these errors.
Official contour/shape APIs: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html .
This implementation uses those APIs and independent NumPy geometry, not copied code.

## Parameters and practical tuning

All defaults are provisional processing-pixel settings. Constructor settings are
available in `model.parameters`; JSON round-trip with `OuterContourDetector(**config)`.

| Parameter | Default | Meaning and increasing-value effect |
|---|---|---|
| `smoothing_px` | `1.0` | Gaussian sigma, px, 0..20. More smooths turn noise but can suppress real sharp/small corners. Raw contour/fit lengths stay unchanged. |
| `corner_window_px` | `8.0` | Chord support arclength on either side, px, 1..100. Larger measures broader bends and needs more perimeter. |
| `corner_turn_deg` | `45.0` | Absolute turn threshold, degrees (0,180]. Larger reports fewer angle events. Zero is straight;90 is a right-angle turn. |
| `min_corner_spacing_px` | `8.0` | Cyclic arclength suppression distance, px, 1..200. Larger removes nearby events, possibly true neighboring corners. |
| `straight_turn_deg` | `5.0` | Local absolute-turn limit for straight-section proposals, degrees 0..180. Larger admits more curved proposals; whole-interval fit guards still apply. |
| `line_error_px` | `1.5` | Maximum raw-point perpendicular TLS line error, px, nonnegative. Larger accepts noisier/curvier lines. This is max error, not RMS. |
| `max_arc_chord_ratio` | `1.15` | Maximum raw arclength/chord ratio for a straight interval, >=1. Larger tolerates more staircase/roughness/backtracking. |
| `min_section_length_px` | `8.0` | Shorter final sections are retained as unknown, nonnegative px. Larger withholds short classifications, not coverage. |
| `min_straight_length_px` | `20.0` | Minimum supported straight interval, nonnegative px. Larger rejects short locally flat patches on curves; small whole straight intervals become unknown. |
| `clip_margin_px` | `1` | Integer0..10, extra inward margin near frame/unobserved domain. Larger marks more boundary evidence uncertain; even0 detects direct artificial edges. |
| `max_depth_jump_m` | `0.02` | Maximum adjacent camera-Z jump for optional complete 3D length. Larger risks joining different surfaces. Not a pixel threshold. |

First verify the mask and observation domain. Set support/smoothing lengths for
resolution, then turn threshold/spacing; inspect curved negatives before loosening
line error. Finally choose target-length/support tolerances. Pixel support parameters
scale with resized images; angle/ratio settings and metric thresholds do not. A high
turn threshold does not guarantee that a noisy bend will become one useful curve.
Short/complex contours can remain unknown. No universal noise-robust preset is claimed.

## Cropping and missing observation

`observed_mask=None` assumes the whole input image is observed. Explicit bool
`observed_mask` describes the valid visible domain; the contour uses object AND
observed. The image border and observed-domain intersections are marked uncertain,
with a margin. The algorithm does not know that a rectangle was artificially cut
from a mask unless the image boundary or supplied observation domain reveals it.
A partial visible side can still contain an independently observed straight interval;
its full original material length is unknown. Supply the original crop/ROI provenance.

Geometry can be computed on clipped sections, but length queries never confirm them.
Do not raise margins to hide segmentation errors or mark an unobserved region as
empty background. Repeated points at self-touching/narrow connections make the
component ambiguous for query confirmation even if some local measurements exist.
Curves separated only by a one-pixel mask bridge are not guaranteed material identity.

## Contour size, largest selection and sequence traversal

Every external contour now includes:

| Field | Meaning |
|---|---|
| `area_px2` | Absolute area enclosed by the raw pixel-center outer polygon, in squared processing pixels. Includes its holes. |
| `area_px` | Existing alias, unchanged; equal to area_px2. |
| `hole_area_px2` | Sum of direct hole contour polygon areas, not the number of background pixels. |
| `foreground_pixel_count` | Exact count of this 8-connected visible object component's pixels after applying observed_mask. Excludes holes and separate foreground islands within holes. |
| `perimeter_px` | Alias of arc_length_px: cyclic raw pixel-center outer perimeter, excluding hole perimeters. |

Each hole also has area_px2 (alias of its existing area_px) and perimeter_px.
Polygon areas and pixel counts have different raster conventions: an inclusive
41-by-21-pixel rectangle has area_px2=800 but foreground_pixel_count=861.
Subtracting hole polygon area is not an exact material pixel count; use the explicit
foreground count. No square-metre area or hidden object size is inferred. Clipped
sizes are only sizes of the observed region and retain the existing clipped flag.

```python
largest = result.largest_contour()  # by='area_px2'
# Alternative: result.largest_contour(by='foreground_pixel_count')
# Alternative: result.largest_contour(by='perimeter_px')
if largest is not None:
    print(largest['id'], largest['area_px2'], largest['foreground_pixel_count'])
    for entry in result.resolve_sequence(largest['id']):
        record = entry['record']
        if entry['type'] == 'segment':
            print(entry['id'], record['kind'], record['fit_length_px'])
        else:  # corner
            print(entry['id'], record['xy'], record['turn_deg'])
```

`largest_contour(*, by='area_px2')` returns a detached deep copy of one complete
contour dictionary, or None for an empty result. Allowed keys are exactly area_px2,
foreground_pixel_count and perimeter_px. Ties choose the smallest integer contour
ID, not the first list position. Unknown keys, duplicate/noninteger IDs or missing,
nonfinite/negative size measurements raise ValueError. The source list is not sorted
or filtered. Hole outlines are never largest-exterior candidates. Choosing the largest
polygon may select a thin ring over a smaller solid component; choose foreground
pixel count when that better matches the intended policy.

`resolve_sequence(contour_id=None)` returns a list of
`{'type': 'segment' or 'corner', 'id': integer, 'record': detached dictionary}`
in exactly the stored cyclic sequence order. Omitted/None selects the largest
polygon-area contour; an empty result returns []. An explicit contour ID must exist.
The helper builds ID maps rather than indexing lists: nonconsecutive integer IDs
work, and corner/segment IDs are separate namespaces. It checks for duplicate record
IDs, missing/unknown references, duplicate references and omitted records; these
raise ValueError rather than silently dropping evidence. It does not reconstruct or
reclassify a damaged sequence. Mutating returned dictionaries does not change the
result. Existing raw records/sequence remain available for callers using them directly.

The helpers do not change query_edge_length's scope: that query still searches all
contours. To associate its intervals with a selected contour, filter the query records
by contour_id. Neither helper changes detection, debug rendering, thresholds or IDs.

## Optional 3D lengths

```python
from depth_image_lens import CameraIntrinsics
result = model.detect(
    object_mask, depth=depth_uint16, depth_scale_m=.001,
    intrinsics=CameraIntrinsics(fx, fy, cx, cy), registered_to_mask=True,
    depth_valid_mask=usable_boundary_depth, diagnostics=True,
)
query_m = result.query_edge_length(.20, unit='m', tolerance=.01)
```

Depth is optical-axis Z, not radial range. Scale is explicit for every numeric dtype:
.001m/count for millimetres,1 for metres. Intrinsics must match this rectified/cropped
pixel grid; no world/robot calibration or table plane is needed for 3D point distances.
`registered_to_mask=True` is a caller declaration, not automatic verification.
Finite positive depth and optional bool `depth_valid_mask` gate samples. Mixed-depth
silhouette pixels may be finite but wrong: exclude them explicitly; no automatic
interior extrapolation or depth-hole filling is performed.

Every contour sample and adjacent pair must be usable and within depth-jump tolerance
for a section's complete `metric.status='measured'`. Otherwise full lengths are null;
`observed_adjacent_length_m` sums only observed continuous pairs and is **not** the
missing full length. Sample/pair validity is returned. Optional malformed/missing
metric evidence preserves all pixel-geometry outputs and records the metric reason.
Fatal object/observation mask contracts still raise. Metric quantities are boundary
point measurements, not verified material length, layer count or grasp safety.

## Output and length query

`OuterContourResult` fields: `contours`, `holes`, `parameters`, `summary`, copied bool
`object_mask`, bool `observed_mask`, optional `diagnostics`. `to_dict()` excludes image
arrays/diagnostics and is strict JSON compatible. Coordinate points and fit endpoints
are XY; arrays index YX. Diagnostic sample coordinates are derived, never raw replacements.

Each contour includes `id`, `status`, `points_xy`, `closed`, `orientation`, `area_px`,
`area_px2`, `hole_area_px2`, `foreground_pixel_count`, `perimeter_px`,
`edge_count`, `arc_length_px`, `clipped`, `self_touch`, `hole_ids`, `segments`, `corners`,
`sequence`, `sequence_start_index`, `coverage_edge_count`. Status is computed,
insufficient_support, ambiguous_self_touch or insufficient_computation_budget.
`area_px` is polygon area of the outer contour, before subtracting holes; it is
not the count of object-mask pixels. A contour needs >=8 raw samples and perimeter >=4 corner windows for supported
classification. Over20,000 raw samples skips turn processing and returns unknown
classification with raw coverage; reduce image resolution for practical use.

Segments include kind straight/curve/unknown, raw-edge index interval, raw XY points,
corner endpoint IDs (null for ordinary joins), visibility observed/uncertain and reasons:

- `arc_length_px`: sum of raw adjacent distances.
- `chord_length_px`: endpoint distance.
- `fit_length_px`: span projected onto the TLS line, not contour arclength.
- `fit_endpoints_xy`, `line_rms_px`, `line_p95_px`, `line_max_error_px`, `arc_chord_ratio`.
- `mean_abs_turn_deg`, `max_abs_turn_deg`: smoothed local window-turn evidence.
- `raw_signed_turn_deg`, `raw_absolute_turn_deg`: summed consecutive raw-edge turns
  inside the section, excluding wrap/endpoint events. Absolute turn can be large
  from raster zigzags; it is a noise measurement, not physical integrated curvature.
- `metric`: status/reason, sample/pair validity, valid_fraction, observed adjacent sum,
  full `arc_length_m`, `chord_length_m`, `fit_length_m`, `line_rms_m`,
  `line_max_error_m` and fitted3D endpoints when usable.

Corners are point events with ID, raw point index/XY, signed turn, convex/concave kind,
and visibility. No extra length is allocated to a corner. `sequence` interleaves
corner events and segment IDs; smooth section joins have no invented corner.

`query_edge_length(target_length, *, unit='px', tolerance=None, mode='match',
min_support_px=20., max_line_error_px=None, max_metric_line_error_m=.005,
require_corner_endpoints=False)` uses
**fit length** of straight intervals. Pixel/m tolerance must be supplied in the same
unit as the target. None uses5px or0.005m according to unit; these defaults are
provisional, so choose a site-specific tolerance. Match tests abs(length-target)<=tolerance. At_least tests
length+tolerance>=target. Comparisons allow floating roundoff of8 machine epsilons
times max(1, length, target, tolerance); this is not a sensor-error allowance.
The returned `length_kind` is explicitly `tls_projected_span`. The query error limit defaults to detector line_error_px;
raising it cannot restore intervals already classified curve/unknown. Metric queries
add `max_metric_line_error_m=.005` (metres): image-straight intervals can curve in3D.
Above this perpendicular3D line error, the metric query remains unknown while pixel
queries and all length measurements remain available. Larger values loosen this
query-only guard; they do not change contour decomposition. Optional
corner-endpoint requirement excludes tangent or artificial section joins.

Returns candidate_found / unknown / no_matching_visible_interval, individual
candidate/not_matched/unknown records and material_edge_identity='unknown'. This
is neither an exact original-side classifier nor proof of absence of a hidden edge.
Uncertain/clipped/ambiguous/metric-missing candidates remain unknown. IDs are local to
this result and can change with parameters, resize or mask changes; no temporal tracking.

## Debug and examples

```python
result.render_debug(color=None, color_order='RGB', depth=None,
                    depth_scale_m=.001, show_raw=True, show_ids=True, panel=True)
result.save_debug('outline.png', color_rgb, color_order='RGB', overwrite=False,
                  show_ids=False)
```

Rendering returns a uint8 image in the requested RGB/BGR order. Source color is
uint8 HxWx3; absent color uses depth percentile display or mask shading. Green is
straight, blue curve, gray short/holes, magenta observation uncertainty, orange
corners (magenta when uncertain). With IDs enabled, corner labels include signed
turn degrees and a white ring marks the cyclic sequence start. Disable IDs for dense shapes. Raw points and fitted spans remain in JSON.
DiagnosticsFalse still renders all known intervals. Saving explicitly returns a
Path, writes PNG only, requires an existing non-symlink parent and refuses overwrite
unless requested. Detection never writes. Source/result arrays are not mutated.

Run `python examples/example_outline.py --output NEW_IMAGE.png`. The example builds
a synthetic mask and queries a known pixel-center side. Dataset-specific replay,
parameter comparison and Japanese human instructions live outside Git; existing
real masks have no verified straight/curve/physical-length labels. They support
operational sensitivity and visual checks, not a real accuracy claim.

## Exact visible component mask

`result.component_mask(contour_id=None)` returns a new bool mask from the visible
8-connected component identified by the contour's raw start point. Omit the ID
for the largest polygon area. Holes and separate objects inside holes stay
excluded; no polygon filling is used. Empty outlines return an empty mask, while
invalid explicit IDs or malformed seed coordinates raise `ValueError`. Recompute
the outline after changing its input mask. See [overhead pick candidates](PICK_CANDIDATES.md)
for a depth-supported interior-region use case. The previous example helper
`contour_component_mask(result, contour_id)` delegates to this installed method.
