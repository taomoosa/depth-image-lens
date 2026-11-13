"""ROI-free top-entry bridge: synthetic masks, never real-camera validation."""
import json
import numpy as np
from depth_image_lens import TopEntryBridgeDetector


def main():
    mask = np.zeros((80, 100), dtype=bool)
    # Two cropped grippers and a lower object connection.
    mask[:40, 15:25] = True
    mask[:40, 70:80] = True
    mask[30:40, 15:80] = True
    depth = np.where(mask, 1.0, 1.5)
    detector = TopEntryBridgeDetector(entry_band_height_px=8,
                                      min_entry_width_px=4,
                                      min_path_width_px=3)
    observed = detector.detect_mask(depth, mask, depth_unit='m', diagnostics=True)
    # Preserve depth validity but simulate missing foreground evidence.
    interrupted = mask.copy()
    interrupted[:, 45] = False
    candidate = detector.detect_mask(depth, interrupted, depth_unit='m', diagnostics=True)
    print(json.dumps({'input': 'synthetic, not real-camera validation',
                      'observed': observed.to_dict(),
                      'interrupted': candidate.to_dict()}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
