"""Conservative optional empty-color veto for externally supplied masks."""
import numpy as np
from .boundary import _background_selection
from .debug import _opencv


def filter_mask_background(color, background, order, candidate, color_valid,
                           check, depth_valid, delta, inferred, bp, parameters):
    bg = np.asarray(background)
    if bg.dtype != np.uint8 or bg.shape != color.shape:
        raise ValueError('background_color must match current uint8 HxWx3')
    rgb = color if order == 'RGB' else color[..., ::-1]
    lab = _opencv().cvtColor(rgb, _opencv().COLOR_RGB2LAB).astype(float)
    cp = dict(bp, background_difference_mode='separate_tolerances',
              background_shadow_mode='off', color_distance=1., color_margin=0.,
              background_chroma_tolerance=parameters['background_match_chroma_tolerance'],
              background_luma_tolerance=parameters['background_match_luma_tolerance'])
    foreground, uncertain, _, info, _ = _background_selection(
        rgb, lab, bg, order, color_valid, check, (0, 0), cp)
    info.pop('fallback', None)
    usable = info['status'] not in ('rejected', 'insufficient')
    matching = candidate & color_valid & ~foreground & ~uncertain if usable else np.zeros_like(candidate)
    # A color match alone is not evidence of zero height. Never suppress invalid,
    # near-threshold, inferred, or clearly raised pixels.
    near_table = depth_valid & (np.abs(delta) < max(0., bp['depth_threshold_m'] - bp['depth_margin_m']))
    removed = matching & near_table & ~inferred
    ambiguous = candidate & (~color_valid | uncertain) if usable else candidate.copy()
    ambiguous |= matching & ~removed
    info['selection_action'] = 'remove_clear_color_matches_with_near_table_depth' if usable else 'keep_external_mask_reference_unusable'
    return candidate & ~removed, dict(background_removed_mask=removed,
        background_matching_mask=matching, background_match_ambiguous_mask=ambiguous), info
