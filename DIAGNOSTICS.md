# Inspecting a detection

Opt in per call; normal JSON output remains unchanged when diagnostics are off:

```python
result = detector.detect_auto(depth, difference, upper_band=(8, 20), diagnostics=True)
# Manual mode also accepts diagnostics=True.
summary = result.to_dict()
arrays = result.diagnostics
```

This option does not change classification. The existing foreground, candidate-region and invalid masks remain on the result. New diagnostic masks/points are allocated only when requested; no image dependency is required by the core.

| Diagnostic | Meaning |
|---|---|
| `candidate_search_mask` | Visible eligible region below the automatic upper band, before foreground filtering. |
| `search_mask` | Manual/inferred anchor corridor used by the final detector, including positions that may be invalid. Compare with `invalid_mask`. |
| `left_anchor_mask`, `right_anchor_mask` | Anchor bands used for component connection tests. |
| `bridge_corridor_mask` | Observed pixels in components reaching both anchors. This is a candidate component, not a unique shortest path or verified material correspondence. Invalid pixels and tolerated gaps are not filled. |
| `sag_path_xy`, `sag_chord_xy` | Observed upper-envelope sample points and anchor chord in image xy coordinates. Missing bins have no sample. |
| `sag_path_used` | Whether the sampled envelope actually produced a sag classification. Points shown when false are diagnostic observations only. |
| `left_drape_component_mask`, `right_drape_component_mask` | Components connected to each anchor. These masks can overlap completely for a single bridge; they are not a left/right segmentation. |
| `opposite_search_mask` | Where an opposite candidate could be searched after one observed neck. It is a search region, not an invented hand. |

Some entries are absent if detection stops before that stage. Reasons and `auto.stage` explain where it stopped. `result.to_dict()` summarizes bool masks by shape and pixel count; point arrays become lists. `include_masks=True` expands bool masks to lists as well. Direct NumPy arrays remain available in `result.diagnostics`.

## Save arrays, JSON and an overlay

After installing the package, from the repository root:

```bash
python examples/example_debug.py --output debug-output
python examples/example_debug.py --depth depth.npy --difference difference.npy --config examples/config.auto.json --output debug-output
```

The no-input example is a synthetic U-shaped deformable-object bridge. The output directory receives:

- `arrays.npz`: foreground/invalid/candidate masks and available diagnostic arrays. Load with `np.load(path, allow_pickle=False)`.
- `result.json`: classifications, reasons, features and compact diagnostic metadata.
- `overlay.svg`: four panels for foreground/invalid pixels, search/bridge/anchors, sag measurement, and anchor-connected drape components.

Orange/blue distinguish the left/right anchor components; green marks a bridge component; red marks invalid depth. Black boxes are unverified hand candidates. The sag panel plots individual observed points, so missing bins are not connected by a fabricated line. SVG generation uses only NumPy and the Python standard library. Large masks may create large SVGs; use a consistent crop before detection. Reusing an output directory replaces these three diagnostic files. Each is written to a fresh temporary file in the same directory and published with `os.replace`: an existing output-file symlink is replaced as a directory entry, and its target is not opened or changed. The output directory itself must not be a symlink. Symlinks in ancestor directories follow normal filesystem resolution; choose a trusted parent directory. This is per-file replacement, not an all-three-files transaction or protection against concurrent hostile parent-directory renames.

These are visible-shape observations, not proof of physical holding or tension. Preserve depth sign, units, registration and invalid-background provenance when inspecting results.

## Shared CLI output

`depth-image-lens` uses the same `depth_image_lens.diagnostics.save_debug(result, output)` function as `examples/example_debug.py`; there is one diagnostic writer. It accepts separate NPY inputs or a numeric NPZ pair through the CLI described in [README.md](README.md). Result JSON never expands masks by default and is serialized with `allow_nan=False`. Only diagnostic arrays are saved in the NPZ; original depth files are not copied. Generated files contain no source paths or arbitrary config metadata.

SVG structure and NPZ/JSON consistency have been checked. A browser rendering check was unavailable in the development environment; no new renderer was installed for this feature.

Continuity diagnostics: the search panel additionally displays `inferred_gap_mask` in yellow and `depth_rejected_mask` in pink. Yellow is explicitly inferred connectivity through known non-foreground pixels, not observed foreground or hole filling. JSON `auto.candidate_evaluations` explains the bounded candidate ranking and rejected depth paths. Sag dots continue to show observed envelope samples only.
