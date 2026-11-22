"""Compare measured oblique outcomes with an opt-in dropout hypothesis."""
import numpy as np

from depth_image_lens import ObjectOnlyGraspDetector, TopEntryBridgeDetector


def main():
    shape = (64, 96)
    cloth = np.zeros(shape, bool)
    cloth[22:30, 16:78] = True
    grippers = np.zeros(shape, bool)
    grippers[:22, 16:26] = True
    grippers[:22, 68:78] = True
    full = cloth | grippers
    yy, xx = np.indices(shape)
    # Measured surfaces on each side are smooth, but their depths differ.
    depth = np.where(full, .95 + .003*xx + .002*yy + .25*(xx >= 47), 1.5)
    difference = np.where(full, .05, 0.)
    for y, x, width in zip(range(22, 30), (46,46,47,47,46,45,45,46), (2,2,3,2,2,3,2,2)):
        depth[y, x:x+width] = 0.  # stepped invalid-depth stripe

    # Deliberately pass depth-cut masks and no separate color evidence.
    measured = depth > 0
    grasp = ObjectOnlyGraspDetector(min_width_px=60, invalid_gap_max_px=3).detect(
        depth, cloth & measured, depth_unit='m', difference=difference,
        diagnostics=True)
    bridge = TopEntryBridgeDetector(gap_radius_px=0, invalid_gap_max_px=3).detect_mask(
        depth, full & measured, depth_unit='m', diagnostics=True)
    print('Observed width and bilateral:', grasp.metrics['visible_width_px'], grasp.bilateral)
    print('Width candidate:', grasp.metrics['invalid_depth_gap'])
    print('Observed bridge and generic geometric candidate:', bridge.bridge, bridge.candidate_bridge)
    print('Missing-depth bridge candidate:', bridge.metrics['invalid_depth_gap'])


if __name__ == '__main__':
    main()
