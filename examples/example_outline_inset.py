"""Inset the whole visible component, even when individual corners are uncertain."""
import argparse
import json

import numpy as np

from depth_image_lens import OuterContourDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offset-px', type=float, default=2.)
    parser.add_argument('--shape', choices=('point-touch', 'rectangle', 'l-shape', 'hole'),
                        default='point-touch')
    parser.add_argument('--output-png', help='Optional raster overlay: gray is source, green is inset')
    args = parser.parse_args()
    if not np.isfinite(args.offset_px) or args.offset_px < 0:
        parser.error('--offset-px must be finite and nonnegative')

    mask = np.zeros((116, 140), bool)
    if args.shape == 'point-touch':
        mask[14:56, 20:62] = True
        mask[56:99, 62:105] = True
    else:
        mask[18:99, 20:115] = True
        if args.shape == 'l-shape':
            mask[54:99, 70:115] = False
        elif args.shape == 'hole':
            mask[38:71, 48:81] = False

    result = OuterContourDetector().detect(mask)
    inset = result.inset_component(args.offset_px)
    summary = dict(inset.summary, parts=[dict(id=part['id'],
        pixels=part['foreground_pixel_count'], polygon_status=part['polygon_status'],
        polygon_reasons=part['polygon_reasons'], outer_vertices=len(part['outer']['points_xy']),
        holes=len(part['holes'])) for part in inset.components])
    print(json.dumps(summary, allow_nan=False, indent=2))

    if args.output_png:
        import cv2
        canvas = np.full((*mask.shape, 3), 24, np.uint8)
        canvas[mask] = (110, 110, 110)
        canvas[inset.inset_mask] = (65, 210, 90)
        if not cv2.imwrite(args.output_png, canvas):
            parser.error('could not write the overlay PNG')


if __name__ == '__main__':
    main()
