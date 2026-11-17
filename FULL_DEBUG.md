# Full diagnostic images

`save_debug_bundle()` complements the compact `render_debug()` and `save_debug()`
views. Existing compact methods keep their interface and behaviour. Use the full
export to inspect recorded masks, numeric evidence, region labels and individual
colour-model diagnostics without the compact view's panel limit.

## Minimal usage

```python
result = segmenter.detect(
    color, depth,
    background_color=background_color,
    background_depth=background_depth,
    color_order="RGB", registered_to_depth=True, depth_scale_m=0.001,
    diagnostics=True,
)
index_path = result.save_debug_bundle(
    "debug_run_001", color,
    columns=6, max_rows=4, tile_width=640, models="all",
)
```

Install the optional `debug` dependency for PNG rendering. By default, use a new
directory for each run; existing destinations are rejected. To update a previous
export explicitly:

```python
result.save_debug_bundle("debug_run_001", color, overwrite=True)
```

With `overwrite=True`, a valid previous `index.json` identifies the managed PNG
pages. New pages and the index replace those outputs; obsolete pages listed in
the previous index are removed. Unrelated files are preserved, including PNGs
not listed by the old index. An existing directory without an index is allowed
if none of the new output names collide with existing files. Invalid manifests,
unsafe page names, symlink paths/managed outputs, and unrelated name collisions
are rejected rather than overwritten.

All rendering finishes in a temporary sibling directory before any existing
bundle changes. Ordinary publication failures restore the previous managed
files from a temporary backup. If rollback itself fails (for example, because
of filesystem failure), the exception identifies retained recovery files; do not
delete them until recovery is complete. A returned index path indicates success.
Updating multiple files is not a crash-atomic transaction: do not read or write
the same bundle concurrently, and use separate run directories when historical
results or recovery after forced termination/power loss are required. Unrelated
files are never backed up, replaced or removed by an export.

`color` must match the detector's processing grid, after resizing and cropping,
not the uncropped camera image. Set `color_order="BGR"` for OpenCV input.
Without color, supply `depth=depth, depth_scale_m=0.001` for a depth source view,
or omit both for a neutral source background.

## Layout

- Defaults: six columns, at most four rows, 24 panels per general page.
- Each image panel is 640 pixels wide by default. A 640 by 400 processing image
  therefore stays at native panel resolution; headings add extra height.
- Masks and color/depth, region, training and decision evidence share general
  pages instead of generating a separate file for every small category.
- Overflow continues on the next page; there is no silent 13-panel cutoff.
- Model-specific maps are grouped by model, with at most three models on a page.
  Additional maps or models continue onto further pages.
- Adjust `columns`, `max_rows` and `tile_width` for smaller displays. These are
  display settings only and do not change any detector parameter or measurement.
  Supported ranges: 1–12 columns, 1–12 rows, 160–2048 pixel panel widths.
  Excessively large pages are rejected to avoid unbounded rendering allocations.
- Use `models=[11, 22]` to select retained diagnostic model IDs (not stack
  positions); filtered model maps remain listed in the index. Unknown IDs fail
  explicitly. `models="all"` is the default.

The JSON index identifies each recorded item and its page and panel position.
It also records unsupported or unavailable items, rather than fabricating images
for data that the detector did not retain. Keep the index with the PNG files.

## Reading the maps

Boolean maps show individual conditions independently. Several rejection
conditions may apply at the same pixel; one reason does not cancel another.
Label maps use categorical colours, not ordered scores. Numeric maps provide
range/unit information where available; their visual scaling is not a new
acceptance threshold. Consult the recorded metadata and parameters for the actual
thresholds. Missing/nonfinite numeric measurements are distinct from finite zero.
Where an evaluation mask is available, unevaluated pixels must not be read as a
successful zero score.

The export uses the retained diagnostics. It cannot reconstruct omitted model
stacks or absent intermediate measurements when detection ran without
`diagnostics=True`. Full diagnostics have a detection-time memory cost; exporting
pages one at a time bounds additional rendering memory but does not remove the
memory already used by the result arrays. Exporting never changes the detection
result or recomputes the decision.

Run `python examples/example_full_debug.py /tmp/lens-full-debug-new` for a
self-contained synthetic RGB-D example with a 640 by 400 image and missing depth.
This checks export mechanics, not segmentation accuracy on a real camera.

Numeric images use finite per-panel scaling. Magenta marks NaN (unavailable or
outside the map's evaluation scope), red marks positive infinity and blue marks
negative infinity. Model source distance is an obstacle-free Chebyshev lower
bound, **not** the actual growth path length; it must not be compared to the
growth budget as an equivalent decision rule. Separate luma/chroma distances
are components of the combined colour metric, not independent acceptance gates.
Rejection bit maps include selected pixels too, and their bit values are recorded
in the index. Full descriptions, parameters and thresholds remain in the index
when they cannot fit inside a panel heading.

## Color-first region evidence

Add `--mode color_first` to the example command to export the weak-depth region
mode. General pages include `first_candidate_mask`, `first_color_group_map`,
`first_region_label_map`, `first_interior_mask`, `first_surround_mask`, the final
`object_mask`, and the acquired depth-statistic/failure maps. Use
`index.json` -> `stages.fusion.regions` for each region ID's statistics, threshold,
acceptance, failures and supported block fraction.

If an extension shares the object's region ID and appears in the final mask but
not the interior mask, it may have been accepted through whole-region inference,
not measured directly. Interior and surround masks are unions across regions.
The failed-block-fraction map marks entire rejected regions, not individual
unsupported blocks. Per-block support and occupancy-excluded blocks are not
retained or reconstructed by this exporter. Compare numeric values in the index,
not brightness between independently scaled panels.

Luma distances are measured in OpenCV's 8-bit encoded Lab L units (255/100 times
L*), not raw CIE L* units. Gate-margin maps have a decision boundary at zero; any
original threshold in their metadata belongs to the underlying measurement.
