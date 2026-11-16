"""Measure a synthetic hanging mask. No real-camera classification is claimed."""
import numpy as np
from depth_image_lens import HangingProfileAnalyzer


def main():
    mask = np.zeros((400, 640), dtype=bool)
    mask[40:140, 140:500] = True  # broad upper material
    mask[140:330, 290:350] = True  # only a narrow tail reaches down
    analyzer = HangingProfileAnalyzer(reach_drop_px=200, upper_band_height_px=100)
    result = analyzer.analyze_mask(
        mask, reference_y=40, expected_x_range=(140, 500),
        observed_mask=np.ones_like(mask),
        mask_complete=True,  # true for this exact synthetic mask, not a real sensor
        diagnostics=True,
    )
    print('Drop quantiles:', result.metrics['drop_quantiles_px'])
    print('Reaching fraction of reference span:', result.metrics['reaching_width_fraction'])
    print('Upper area fraction:', result.metrics['upper_area_fraction'])
    print('Supported bottom array:', result.profiles['bottom_y_px'])
    # With OpenCV installed: rgb_debug = result.render_debug()
    # For real data, retain NaN/censored flags rather than filling missing columns.


if __name__ == '__main__':
    main()
