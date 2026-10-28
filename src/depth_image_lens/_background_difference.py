"""Independent encoded-Lab tolerances; scores are distances, not probabilities."""
import numpy as np


def separate_lab_selection(delta, available, clipped, parameters):
    """Return anisotropic background difference with finite saturated diagnostics."""
    tc = parameters['background_chroma_tolerance']
    tl = parameters['background_luma_tolerance']
    margin = parameters['background_score_margin']
    chroma = np.hypot(delta[..., 1], delta[..., 2])
    luma = np.abs(delta[..., 0])
    # Very small valid tolerances can overflow division. Saturation preserves the
    # >1 decision while keeping numeric maps finite on available pixels.
    with np.errstate(over='ignore', divide='ignore', under='ignore'):
        cr = np.minimum(chroma / tc, 1e6)
        lr = np.minimum(luma / tl, 1e6)
    radius = np.minimum(np.hypot(cr, lr), 1e6)
    if tc == tl:
        distance = np.linalg.norm(delta, axis=2)
        with np.errstate(over='ignore', divide='ignore', under='ignore'):
            radius = np.minimum(distance / tc, 1e6)
        foreground = available & (distance > tc)
        near = np.abs(distance - tc) <= margin * tc
    else:
        foreground = available & (radius > 1.)
        near = np.abs(radius - 1.) <= margin
    uncertain = ~available | near | (available & clipped)
    score = radius ** 2
    saturated = available & (radius >= 1e6)
    maps = dict(background_chroma_distance=chroma.copy(),
                background_luma_distance=luma.copy(), background_combined_score=score,
                background_score_saturated_mask=saturated,
                background_photometric_clipped_mask=available & clipped)
    for key in ('background_chroma_distance', 'background_luma_distance', 'background_combined_score'):
        maps[key][~available] = np.nan
    info = dict(mode='separate_tolerances', chroma_tolerance=tc, luma_tolerance=tl,
                units='OpenCV encoded 8-bit Lab; not standardized Delta E',
                foreground_rule='(dC/Tc)^2 + (dL/Tl)^2 > 1', score_threshold=1.,
                uncertainty_radius_margin=margin, score_saturation=1e12,
                saturated_pixels=int(saturated.sum()), clipped_pixels=int((available & clipped).sum()),
                warnings=['distance_score_is_not_probability_or_physical_certainty',
                          'larger_luma_tolerance_can_hide_brightness_only_objects',
                          'shadow_and_brightness_only_object_can_be_identical',
                          'reference_check_thresholds_remain_unweighted_Lab',
                          'clipped_pixels_remain_uncertain'])
    return foreground, uncertain, info, maps
