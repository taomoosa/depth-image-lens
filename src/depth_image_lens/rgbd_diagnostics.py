"""Separate missingness evidence from appearance uncertainty; NumPy only."""
from dataclasses import replace
import numpy as np
from .bridge_drape import _mask


def uncertainty_report(depth, difference, result, colors, *, observed_mask=None,
                       background_valid=None, color_valid_mask=None, arm_mask=None):
    """Return JSON counts/locations plus overlapping cause masks, without inference.

    Zero difference cannot distinguish background from caller exclusion. Masks
    identify co-occurring evidence, not a counterfactual proof of causation.
    Input arrays use the 16UC1 convention. Result must have diagnostics enabled.
    """
    depth, difference = np.asarray(depth), np.asarray(difference)
    if depth.dtype != np.uint16 or difference.dtype != np.uint16 or depth.shape != difference.shape or depth.ndim != 2:
        raise ValueError('same-grid uint16 depth and difference required')
    shape = depth.shape
    def supplied(mask, name):
        return np.ones(shape, bool) if mask is None else _mask(mask, shape, name)
    zero = np.zeros(shape, bool)
    valid = (depth > 0) & supplied(observed_mask, 'observed_mask') & supplied(background_valid, 'background_valid')
    causes = {
        'sensor_zero_depth': depth == 0,
        'caller_unobserved': ~supplied(observed_mask, 'observed_mask'),
        'background_unavailable': ~supplied(background_valid, 'background_valid'),
        'zero_difference_background_or_excluded': (difference == 0) & valid,
        'caller_arm_excluded': zero.copy() if arm_mask is None else _mask(arm_mask, shape, 'arm_mask'),
        'color_unavailable': ~supplied(color_valid_mask, 'color_valid_mask') & result.foreground_mask,
        'appearance_ambiguous': colors.appearance_uncertain_mask & valid & supplied(color_valid_mask, 'color_valid_mask'),
        'color_boundary_guard_only': colors.uncertain_mask & ~colors.appearance_uncertain_mask,
    }
    diag = result.diagnostics
    regions = {'left_anchor': diag['left_anchor_mask'], 'right_anchor': diag['right_anchor_mask'],
               'search': diag.get('search_mask', zero), 'observed_bridge_component': diag.get('bridge_corridor_mask', zero)}
    regions['outside_search'] = ~regions['search']
    records = {}
    for name, mask in causes.items():
        points = np.argwhere(mask)
        records[name] = {'pixels': int(mask.sum()), 'bbox_xyxy':
            [int(points[:, 1].min()), int(points[:, 0].min()), int(points[:, 1].max()+1), int(points[:, 0].max()+1)] if len(points) else None,
            'regions': {key: int((mask & region).sum()) for key, region in regions.items()}}
    return {'causes': records, 'output_reasons': dict(result.reasons),
            'causes_overlap': True, 'inferred_depth_pixels': 0,
            'zero_difference_semantics': 'background or mask exclusion; indistinguishable without separate provenance'}, causes


def guard_masked_negative_claims(result, difference, *, verified_nonforeground_mask=None):
    """Opt-in guard for absence claims from a clipped/masked difference image.

    A zero is not evidence that masking preserved every relevant object pixel.
    The caller may attest zero pixels as measured background or independently
    verified irrelevant exclusions. Without that provenance, negative bridge
    and one-sided/neither drape claims become unknown. Positive paths and both
    observed drapes are unchanged. This never recovers a removed path or contact.

    Use a diagnostics-enabled result. Defaults in the detector are unchanged.
    An incorrect caller attestation cannot be detected by this function.
    """
    delta = np.asarray(difference)
    shape = result.foreground_mask.shape
    if delta.shape != shape or delta.dtype != np.uint16:
        raise ValueError('same-grid uint16 difference required')
    if 'search_mask' not in result.diagnostics:
        raise ValueError('a result with search_mask diagnostics is required')
    verified = np.zeros(shape, bool) if verified_nonforeground_mask is None else _mask(
        verified_nonforeground_mask, shape, 'verified_nonforeground_mask')
    search = result.diagnostics['search_mask']
    unresolved = search & (delta == 0) & ~result.invalid_mask & ~verified
    bridge, drape = result.bridge, result.drape
    reasons = dict(result.reasons)
    changed = []
    if unresolved.any():
        if bridge == 'no':
            bridge = 'unknown'; changed.append('bridge')
            reasons['bridge'] = 'unverified_zero_difference_cannot_establish_absence'
        if drape in ('left', 'right', 'neither'):
            drape = 'unknown'; changed.append('drape')
            reasons['drape'] = 'unverified_zero_difference_cannot_establish_opposite_absence'
    features = {**result.features,
                'negative_evidence_guard_applied': True,
                'negative_evidence_before_guard': {'bridge': result.bridge, 'drape': result.drape},
                'unverified_zero_difference_search_pixels': int(unresolved.sum()),
                'negative_evidence_changed_outputs': changed,
                'nonforeground_provenance': 'caller_attested' if verified_nonforeground_mask is not None else 'not_supplied',
                'negative_evidence_inferred_pixels': 0}
    diagnostics = {**result.diagnostics, 'unverified_nonforeground_mask': unresolved}
    stages=dict(result.stage_results)
    stages['negative_provenance']={'status':'computed','unverified_pixels':int(unresolved.sum()),
        'changed_outputs':changed,'before_guard':features['negative_evidence_before_guard']}
    if 'bridge' in changed and 'observed_bridge' in stages:
        stages['observed_bridge']={**stages['observed_bridge'],'value':'unknown',
            'reason':reasons['bridge']}
    return replace(result, bridge=bridge, drape=drape, reasons=reasons, features=features, diagnostics=diagnostics,stage_results=stages)
