"""Remove table leakage from a supplied mask using an aligned empty RGB image."""
import numpy as np
from depth_image_lens import OverheadObjectCounter


def main():
    empty_color = np.full((60, 80, 3), (150, 140, 100), np.uint8)
    color = empty_color.copy()
    color[15:40, 20:50] = (210, 35, 35)
    empty_depth = np.full((60, 80), 1000, np.uint16)
    depth = empty_depth.copy()
    depth[15:40, 20:50] = 980
    padded_mask = np.zeros((60, 80), bool)
    padded_mask[11:44, 16:54] = True
    counter = OverheadObjectCounter(background_match_chroma_tolerance=8.,
                                    background_match_luma_tolerance=12.)
    result = counter.detect_mask(
        color, padded_mask, color_order='RGB', background_color=empty_color,
        depth=depth, background_depth=empty_depth, depth_scale_m=.001,
        registered_to_depth=True, diagnostics=True)
    print('count:', result.count, 'candidates:', result.candidate_count)
    print('removed pixels:', result.metrics['background_removed_pixels'])
    print('reasons:', result.reasons)
    # Inspect result.diagnostics['background_removed_mask'] on real inputs.
    # Same-color thin objects cannot reliably be distinguished from the table.
    return result


if __name__ == '__main__':
    main()
