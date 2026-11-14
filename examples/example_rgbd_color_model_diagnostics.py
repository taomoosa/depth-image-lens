"""Inspect a rejected pixel without changing segmentation parameters."""
import numpy as np
from depth_image_lens import RGBDBackgroundSegmenter


def explain_pixel(result, y, x):
    """Return independent model tests, not one fabricated winning explanation."""
    d = result.diagnostics
    definitions = result.stages['fusion']['model_rejection_flags']
    report = dict(selected=bool(result.object_mask[y, x]),
                  decision=int(d['fusion_decision'][y, x]),
                  raw_farther=bool(d['model_raw_farther_mask'][y,x]),
                  farther_excluded=bool(d['model_farther_excluded_mask'][y,x]),
                  background_distance=float(d['model_background_distance'][y, x]),
                  background_luma_distance=float(d['background_luma_distance'][y,x]),
                  background_chroma_distance=float(d['background_chroma_distance'][y,x]),
                  metric=result.stages['fusion']['model_color_metric'],
                  models=[])
    for i, model_id in enumerate(d['model_diagnostic_ids']):
        flags = int(d['model_all_rejection_flags'][i, y, x])
        report['models'].append(dict(
            id=int(model_id), color_distance=float(d['model_all_color_distances'][i,y,x]),
            luma_distance=float(d['model_all_luma_distances'][i,y,x]),
            chroma_distance=float(d['model_all_chroma_distances'][i,y,x]),
            palette_index=int(d['model_all_nearest_palette_indices'][i,y,x]),
            source_distance_px=float(d['model_all_source_distances_px'][i,y,x]),
            reachable=bool(d['model_all_reachable_masks'][i,y,x]),
            selected_before_area_filter=bool(d['model_all_selected_masks'][i,y,x]),
            failed_tests=[name for bit,name in definitions.items() if flags & int(bit)]))
    # An empty list is not evidence that the pixel passed any model.
    report['model_records'] = result.stages['fusion']['models']
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Optional debug PNG path; includes spatial rejection tiles')
    parser.add_argument('--model-id', type=int, help='Show one evaluable model instead of ANY model')
    args = parser.parse_args()
    background = np.full((60,100,3),140,np.uint8)
    color = background.copy(); color[10:45,10:40]=80
    color[10:45,40:65]=(20,150,210)
    depth = np.full((60,100),1000,np.uint16)
    reference_depth = depth.copy(); depth[10:45,10:40]=980
    result = RGBDBackgroundSegmenter(fusion_mode='color_model').detect(
        color,depth,background_color=background,background_depth=reference_depth,
        depth_scale_m=.001,color_order="RGB",registered_to_depth=True,diagnostics=True)
    print(explain_pixel(result,25,45))
    if args.output:
        import cv2
        debug = result.render_debug(color, color_order='RGB', model_diagnostic_id=args.model_id)
        if not cv2.imwrite(args.output, debug[..., ::-1]):
            raise OSError('Could not write debug image')


if __name__ == '__main__':
    main()
