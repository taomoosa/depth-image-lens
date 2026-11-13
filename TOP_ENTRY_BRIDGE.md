# Two top-entry bridge detection

`TopEntryBridgeDetector` tests visible connectivity below two broad regions
entering at the image top. It requires **no per-gripper ROI or x coordinate**.
The input boolean mask may contain both grippers and the suspended object.
It does not identify grippers semantically or prove physical grasp.

```python
from depth_image_lens import TopEntryBridgeDetector

result = TopEntryBridgeDetector(
    entry_band_height_px=8,
    min_entry_width_px=4,
    min_path_width_px=3,
).detect_mask(
    depth_m, foreground_mask,
    depth_unit="m",
    observed_mask=depth_observed_mask,
    mask_complete=False,
    diagnostics=True,
)
print(result.bridge, result.candidate_bridge, result.reasons)
```

Depth and masks must be aligned HxW arrays. Declare `depth_unit="mm"` for
millimetres. Color/background depth belong to the upstream mask generation;
the detector does not directly consume them. An `ObjectOnlyGraspDetector`
result's `diagnostics["elevated_mask"]` can be passed, but that class **requires**
an input object mask and does not generate a semantic object/gripper mask.
If that mask excludes grippers, this detector may lack its two top entries.
Do not reduce the input to the largest component or the two largest components.

## Meaning of results

- `bridge="yes"`: two distinct broad top entries have an observed,
  depth-continuous, sufficiently thick connection below the entry band.
- `bridge="unknown", candidate_bridge=True`: a connection exists geometrically,
  but is too thin, crosses invalid/uncertain depth or a depth jump, or requires
  small-gap inference. This is **not** a confirmed bridge.
- `bridge="unknown", candidate_bridge=False`: no supported candidate was found,
  or two unambiguous top entries could not be identified. This is not absence.
- `bridge="no"`: no observed or small-gap candidate connects two entries in a
  complete search. This requires the caller's `mask_complete=True`, valid depth
  throughout the search, no excluded pixels there, and no lower/side clipping.
  Do not attest completeness merely to force a binary output.

## Parameters to tune

1. `entry_band_height_px=8`: number of rows from image row zero used to locate
   entries. Keep this band inside the separately visible grippers and above the
   joining material. No separate ROIs are needed, but this band is an assumption.
   A horizontal bridge inside the band merges the entries and yields unknown.
2. `min_entry_width_px=4`: minimum foreground count per supported band row.
   At least `min_entry_row_fraction=0.6` of rows must meet it; a component must
   reach both the top and bottom of the band. This removes small top specks.
3. `min_entry_gap_px=2`: minimum clear horizontal gap between entry extents.
   Closer or merged entries remain unknown instead of inventing two grippers.
4. `min_path_width_px=3`: odd square support size. Erosion removes thin paths
   within each depth-connected component, then checks the remaining connectivity.
   Increasing to 5 rejects more thin noise
   but also rejects real thin cloth. Setting 1 disables this thickness guard.
   This is pixel support, not calibrated physical width.
5. `min_bridge_drop_px=3`: forbid crossing the central inter-entry gap until this
   many rows below the entry band. If the gripper tips have a known approximate
   y, increasing this to search below them reduces gripper-to-gripper shortcuts.
   Increasing it can also hide real near-tip bridges.
6. `depth_jump_m=0.05`: maximum local adjacent depth change on an observed path,
   always in metres. Too small fragments noisy cloth; too large allows unrelated
   overlapping surfaces to connect.
7. `min_area=20`: minimum observed component size. This includes entry pixels,
   so it is **not** a minimum connector area. The width guard rejects filaments.
8. `gap_radius_px=1`: square closing radius (0 disables; maximum 3) for a
   **candidate only**. It may connect small gaps or nearby separate objects;
   it never supplies new measured depth or changes a result to yes.

The search extends below the top band and horizontally one entry-width outside
the two outer entry edges. Connections outside this window are not claimed.
Side/bottom clipping withholds negative results.

## Missing data and repair

Use `observed_mask=False` for unavailable depth and `exclude_mask=True` for
regions that must never participate. Pass repaired/inferred mask pixels via
`uncertain_mask=True`; these cannot certify an observed connection. If upstream
repair has invented a connection but its provenance is omitted, this detector
cannot distinguish the invention from a measured mask.

A hole is harmless when a thick observed route remains around it. A complete
cut through the bridge produces unknown, even if small-gap geometry reconnects
it. Isolated small components away from the route do not affect the result.
Noise that forms a broad, depth-consistent route can still mimic a real bridge.
Similarly, touching grippers or another connecting object can look identical;
use a below-tip search, upstream separation or additional observations to
resolve those cases. Image geometry alone does not prove grasp or object identity.

With `diagnostics=True`, inspect `entry_labels`, `entry_seed_masks`, `search_mask`,
`observed_path_mask`, `robust_path_core_mask`, `depth_rejected_edge_mask`, and,
when no observed positive exists, `candidate_path_mask` and `inferred_gap_mask`.
Early unknown results may omit later-stage diagnostics. All diagnostics are
returned arrays, not files. `to_dict()` omits them for concise JSON.

Run `python examples/example_top_entry_bridge.py` for a synthetic usage example.
Synthetic tests verify behavior, not real-camera accuracy.
