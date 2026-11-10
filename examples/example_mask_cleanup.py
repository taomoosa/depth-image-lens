"""Synthetic mask-repair comparisons and an explicit RGB-D -> outline pipeline.

Run from an installed checkout: python examples/example_mask_cleanup.py
No files, network requests or robot commands are produced. These invented scenes
check geometric behavior and failure modes, not real-camera accuracy.
"""
import json
import numpy as np
from depth_image_lens import cleanup_mask


def _rectangle():
    mask = np.zeros((64, 80), bool)
    mask[12:52, 12:68] = True
    return mask


def synthetic_cases():
    """Each tuple is (name, truth, input, settings); truth is synthetic only."""
    truth = _rectangle()
    damaged = truth.copy()
    damaged[22:24, 22:24] = False
    damaged[34:37, 45:48] = False
    yield 'small_segmentation_holes', truth, damaged, dict(max_hole_area=9)

    jagged = truth.copy()
    jagged[12, 18:62:4] = False
    jagged[11, 20:64:4] = True
    yield 'jagged_exterior', truth, jagged, dict(smoothing_radius_px=1)
    combined = jagged & damaged
    combined[11] = jagged[11]
    yield 'holes_and_jagged_exterior', truth, combined, dict(max_hole_area=9, smoothing_radius_px=1)

    real_hole = truth.copy()
    real_hole[30:32, 35:37] = False
    yield 'real_small_hole_is_indistinguishable', real_hole, real_hole, dict(max_hole_area=9)
    large_hole = truth.copy()
    large_hole[25:38, 30:50] = False
    yield 'large_real_hole_preserved', large_hole, large_hole, dict(max_hole_area=9, smoothing_radius_px=1)

    adjacent = truth.copy()
    adjacent[:, 39:41] = False
    yield 'neighbor_objects_do_not_merge', adjacent, adjacent, dict(smoothing_radius_px=2)
    thin = truth.copy()
    thin[32, 68:76] = True
    yield 'thin_appendage_can_shorten', thin, thin, dict(smoothing_radius_px=1)
    yield 'sharp_corners_can_round', truth, truth, dict(smoothing_radius_px=1)
    protected_corners = np.zeros(truth.shape, bool)
    for y in (12, 51):
        for x in (12, 67):
            protected_corners[y, x] = True
    yield 'protected_corners_preserved', truth, truth, dict(smoothing_radius_px=1, protected_mask=protected_corners)

    cropped = truth.copy()
    cropped[:12, 12:68] = True
    yield 'cropped_frame_not_extended', cropped, cropped, dict(max_hole_area=9, smoothing_radius_px=1)
    open_gap = truth.copy()
    open_gap[12:35, 38:41] = False
    yield 'open_gap_not_reconstructed', truth, open_gap, dict(max_hole_area=100, smoothing_radius_px=1)
    unknown = np.ones(truth.shape, bool)
    unknown[22:24, 22:24] = False
    yield 'unknown_hole_not_filled', truth, damaged, dict(max_hole_area=9, observed_mask=unknown)
    deferred = np.zeros(truth.shape, bool)
    deferred[34:37, 45:48] = True
    yield 'deferred_hole_not_filled', truth, damaged, dict(max_hole_area=9, protected_mask=deferred)
    yield 'missing_object_not_invented', truth, np.zeros_like(truth), dict(max_hole_area=9, smoothing_radius_px=1)
    empty = np.zeros_like(truth)
    yield 'empty_mask_stays_empty', empty, empty, dict(max_hole_area=9, smoothing_radius_px=1)


def _component_count(mask):
    # Deliberately independent of the implementation's run-length labeler.
    remaining = set(map(tuple, np.argwhere(mask)))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (y + dy, x + dx)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
    return count


def _iou(truth, mask):
    union = int((truth | mask).sum())
    return float((truth & mask).sum() / union) if union else 1.0


def run_experiments():
    rows = []
    for name, truth, raw, settings in synthetic_cases():
        result = cleanup_mask(raw, **settings)
        cleaned = result.object_mask
        rows.append(dict(case=name, raw_iou=_iou(truth, raw), cleaned_iou=_iou(truth, cleaned),
                         raw_error_pixels=int((truth != raw).sum()),
                         cleaned_error_pixels=int((truth != cleaned).sum()),
                         added_pixels=result.metrics['added_pixels'],
                         removed_pixels=result.metrics['removed_pixels'],
                         false_foreground_added=int((result.added_mask & ~truth).sum()),
                         true_foreground_removed=int((result.removed_mask & truth).sum()),
                         raw_components=_component_count(raw), cleaned_components=_component_count(cleaned)))
    return rows


def rgbd_pipeline_example():
    """Opt-in postprocessing; the segmenter and its source evidence stay intact."""
    from depth_image_lens import RGBDBackgroundSegmenter, OuterContourDetector
    truth = _rectangle()
    empty_color = np.full((*truth.shape, 3), 190, np.uint8)
    color = empty_color.copy()
    color[truth] = 100
    color[22:24, 22:24] = 190  # A toy segmentation defect, not missing depth.
    background_depth = np.full(truth.shape, 2.0)
    depth = background_depth.copy()
    depth[truth] -= .002  # Below the .005 m segmentation threshold.
    segmented = RGBDBackgroundSegmenter(background_luma_tolerance=30).detect(
        color, depth, background_color=empty_color, background_depth=background_depth,
        color_order='RGB', registered_to_depth=True, depth_scale_m=1.0)
    protected = segmented.deferred_mask | segmented.suppressed_mask
    repaired = cleanup_mask(segmented.object_mask, max_hole_area=4, smoothing_radius_px=0,
                            observed_mask=segmented.outline_observed_mask,
                            protected_mask=protected)
    # Keep uncertainty, inferred geometry and sensor availability separate.
    combined_uncertain = segmented.uncertain_mask | repaired.changed_mask
    outline = OuterContourDetector().detect(
        repaired.object_mask, observed_mask=segmented.outline_observed_mask)
    # This outline describes the repaired hypothesis, not a verified material edge.
    # Use segmented.object_mask for evidence-only measurement. The outline API
    # does not ingest combined_uncertain, so retain it alongside outline output.
    return dict(cleanup=repaired.to_dict(),
                raw_object_pixels=int(segmented.object_mask.sum()),
                combined_uncertain_pixels=int(combined_uncertain.sum()),
                outline_summary=outline.summary,
                scope='repaired_mask_geometry_not_additional_sensor_evidence')


if __name__ == '__main__':
    print(json.dumps(dict(synthetic_only=True, comparisons=run_experiments(),
                          rgbd_pipeline=rgbd_pipeline_example()), indent=2, allow_nan=False))
