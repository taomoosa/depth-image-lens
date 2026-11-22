"""Opt-in gate diagnostics. Overlapping failures are not causal attribution."""
import numpy as np


def detailed_fusion_diagnostics(debug, p, roi, valid, color_fg, depth_fg,
                                color_usable, color_uncertain, farther,
                                selected, filtered):
    """Add inspectable gate maps without feeding any data back into selection.

    Reasons name failed conditions, not a claim that changing one parameter
    would select that pixel. NaN denotes a gate not evaluated at that location.
    """
    out = {}
    meta = dict(debug.get('diagnostic_metadata', {}))
    mode = p['fusion_mode']

    def mask(name, values, description):
        key = 'reason_' + name + '_mask'
        out[key] = roi & values
        meta[key] = dict(kind='overlapping_condition', description=description,
                         legend='Highlighted = condition holds; dark = false. Conditions can overlap.')

    def margin(name, values, threshold, scope, unit, comparison='>='):
        key = 'gate_' + name + '_margin'
        out[key] = np.where(scope, values - threshold, np.nan)
        meta[key] = dict(kind='gate_margin', unit=unit, threshold=float(threshold),
                         comparison=comparison,
                         description='Value minus threshold. Positive passes; zero follows comparison; NaN = outside gate scope.')

    mask('depth_unavailable', ~valid, 'Current/reference depth unavailable or excluded by observation masks.')
    mask('reference_color_unavailable', ~color_usable, 'Reference color evidence unavailable, excluded, or rejected.')
    mask('color_uncertain', color_uncertain, 'Color comparison is uncertain, not necessarily rejected.')
    mask('farther_than_background', farther, 'Raw farther-than-background condition; some modes can override it.')
    mask('removed_by_min_object_area', filtered, 'Selected by fusion but removed by final component-area filtering.')
    out['fusion_final_selected_mask'] = selected.copy()
    meta['fusion_final_selected_mask'] = dict(kind='selection', description='Final selection after minimum-area filtering.')
    if 'fusion_decision' in debug:
        # Preserve every final reason separately instead of hiding categories
        # in a single grayscale image. See stage_results for code meanings.
        for code in np.unique(debug['fusion_decision']):
            mask('decision_' + str(int(code)), debug['fusion_decision'] == code,
                 'Final exclusive fusion decision code ' + str(int(code)) + '; see fusion stage reason legend.')
    delta = debug['depth_difference_m']
    margin('depth_foreground', delta, p['depth_threshold_m'], valid, 'm', '>')
    margin('background_color_difference', debug['background_combined_score'], 1., color_usable, 'normalized squared Lab distance', '>')
    if mode == 'depth_required':
        mask('depth_required_not_foreground', color_fg & ~depth_fg, 'Color-only change does not satisfy depth_required selection.')
    if 'depth_support_fraction' in debug:
        margin('depth_neighbor_support', debug['depth_support_fraction'], p['depth_support_fraction'], depth_fg, 'fraction')
        mask('insufficient_depth_neighbor_support', depth_fg & ~debug['locally_supported_depth_mask'], 'Foreground depth lacks required neighboring support.')
    if mode == 'evidence':
        margin('strong_depth', delta / p['depth_threshold_m'], p['strong_depth_ratio'], depth_fg, 'ratio')
    if mode in ('color_seeded', 'color_supported'):
        relaxed = debug['relaxed_color_mask']
        domain = debug['growth_candidate_mask']
        mask('outside_growth_domain', (color_fg | depth_fg | relaxed) & ~domain, 'Candidate fails one or more growth-domain conditions; seeds may bypass this gate.')
        mask('growth_not_selected', domain & ~selected & ~filtered, 'Inside growth domain but not finally selected: connectivity, color/path budget or later support veto may apply.')
        if mode == 'color_seeded':
            from ._rgbd_fusion import _neighbors8
            adjacent = _neighbors8(selected | filtered)
            mask('growth_compatible_next_step', domain & ~selected & ~filtered & adjacent, 'Could be reached by one more candidate step from final selection; configured step budget ended.')
            mask('growth_no_selected_neighbor', domain & ~selected & ~filtered & ~adjacent, 'Candidate has no selected neighbor; not proof of global disconnection.')
            margin('strict_chroma', debug['background_chroma_distance'] / p['background_chroma_tolerance'], 1 + p['background_score_margin'], color_usable, 'ratio', '>')
        else:
            mask('seed_area_filtered', debug['raw_supported_seed_mask'] & ~debug['supported_seed_mask'] & ~debug['support_rejected_mask'] & ~filtered, 'Raw seed absent from retained seeds (seed-area gate); whole-region support veto is shown separately.')
            mask('region_support_veto', debug['support_rejected_mask'], 'Grown component rejected by measured-depth fraction.')
            mask('background_guard', debug['background_guard_mask'], 'Near-table depth and unchanged reference appearance block growth.')
            margin('region_depth_support', debug['region_depth_support_fraction'], p['supported_min_depth_fraction'], debug['growth_step'] >= 0, 'fraction')
    if mode in ('color_local', 'color_model'):
        candidate = debug['local_candidate_mask']
        domain = candidate & valid & ~debug['local_upper_excluded_mask'] & ~farther
        margin('local_coverage', debug['local_valid_fraction'], p['local_min_valid_fraction'], candidate, 'fraction')
        margin('local_effective_samples', debug['local_effective_samples'], 3., candidate, 'weighted samples')
        margin('local_mean_depth', debug['local_mean_depth_m'], p['local_mean_depth_m'], domain, 'm', '>')
        margin('local_depth_density', debug['local_depth_support_fraction'], p['local_min_depth_fraction'], domain, 'fraction')
        mask('local_upper_excluded', debug['local_upper_excluded_mask'], 'Depth exceeds configured reference separation.')
        mask('local_coverage_failed', candidate & (debug['local_valid_fraction'] < p['local_min_valid_fraction']), 'Insufficient weighted valid-depth coverage.')
        mask('local_samples_failed', candidate & (debug['local_effective_samples'] < 3.), 'Fewer than three effective weighted samples.')
        mask('local_mean_failed', domain & ~(debug['local_mean_depth_m'] > p['local_mean_depth_m']), 'Local mean fails seed gate (not necessarily final rejection).')
        mask('local_density_failed', domain & (debug['local_depth_support_fraction'] < p['local_min_depth_fraction']), 'Local positive-depth fraction fails seed gate.')
        mask('local_seed_area_filtered', debug['raw_local_seed_mask'] & ~debug['local_seed_mask'] & ~filtered, 'Raw local seed component below minimum seed area.')
        mask('locally_table_like', debug['local_background_mask'], 'Local statistics classify as table-like; model mode can override.')
    if mode == 'color_model':
        descriptions = {
            'model_training_mask': 'Legacy view: training pixels that also survive final selection, not all learning input.',
            'model_training_input_mask': 'Raw strong-depth input in processed seed regions, before training guard and final selection; excludes model_max_regions budget omissions.',
            'model_training_used_mask': 'Pixels used to fit accepted palettes, before final selection or area filtering.',
            'model_training_excluded_mask': 'Raw palette input not used by an accepted palette (may include insufficient model or color budget).',
            'model_training_interior_mask': 'Observed color interiors evaluated by color_core; not necessarily accepted.'}
        for key, description in descriptions.items():
            meta[key] = dict(description=description)
        for key in debug:
            if key.startswith('model_training_') and key.endswith('_mask') and key not in descriptions:
                meta[key] = dict(description='Color-core training group condition: ' + key[15:-5] + '. Multiple conditions can overlap; see group statistics.', kind='overlapping_condition')
    if mode == 'depth_anchored':
        mask('refinement_negative', debug['refinement_excluded_mask'] | debug['background_anchor_mask'] | debug['raw_background_guard_mask'], 'Outside refinement domain or blocked by negative/background evidence.')
        mask('refinement_score_failed', debug['refinement_domain_mask'] & (debug['refinement_score'] <= .5) & ~debug['foreground_anchor_mask'], 'Mutable refinement score does not exceed 0.5; fixed anchors are separate.')
        margin('refinement_score', debug['refinement_score'], .5, debug['refinement_domain_mask'], 'heuristic score', '>')
    if mode == 'color_region':
        assigned = debug['region_label_map'] > 0
        for name, key, threshold, unit, comparison in (
                ('area', 'region_area_px', p['region_min_area'], 'pixels', '>='),
                ('valid_fraction', 'region_valid_fraction', p['region_min_valid_fraction'], 'fraction', '>='),
                ('mean_depth', 'region_clipped_mean_depth_m', p['region_mean_depth_ratio'] * p['depth_threshold_m'], 'm', '>'),
                ('depth_fraction', 'region_depth_support_fraction', p['region_min_depth_fraction'], 'fraction', '>=')):
            margin('region_' + name, debug[key], threshold, assigned, unit, comparison)
            for state in ('pass', 'fail'):
                meta['region_' + state + '_' + name + '_mask'] = dict(
                    description='Region ' + name + ' gate ' + state + '. Independent tests may overlap.',
                    threshold=float(threshold), unit=unit, comparison=comparison)
        meta['region_group_color_distance'] = dict(unit='OpenCV Lab', threshold=p['region_color_tolerance'], comparison='<=', description='Distance to assigned fixed representative; NaN means unassigned.')
        for name, description in {
                'region_color_budget_exceeded_mask': 'Color candidate unassigned after the maximum color-group budget.',
                'region_invalid_depth_blocked_mask': 'Region accepted, but this pixel has unavailable depth.',
                'region_farther_blocked_mask': 'Region accepted, but this pixel is farther than the background.',
                'region_accepted_mask': 'Region passes all region gates, before per-pixel depth veto and final area filtering.',
                'region_label_map': 'Connected color-region IDs. IDs identify regions, not scores.',
                'region_color_group_map': 'Fixed color-group IDs. One group may contain multiple disconnected regions.',
                'region_fail_observed_mask': 'Assigned region has no observed depth samples.',
                'region_pass_observed_mask': 'Assigned region has at least one observed depth sample.'}.items():
            meta[name] = dict(description=description)
    if mode == 'color_first':
        for key, description in {
            'first_candidate_mask': 'Color-first candidates before any depth gate; uses first_color_relaxation_ratio.',
            'first_interior_mask': 'Eroded candidate interiors used for signed depth statistics; not final boundary erosion.',
            'first_surround_mask': 'Union of per-region reference rings; excludes every changed-color candidate.',
            'first_insufficient_mask': 'Insufficient region or surrounding-reference evidence; remains unknown.',
            'first_upper_excluded_mask': 'Depth exceeds local_max_reference_separation_m; never included.',
        }.items():
            meta[key] = dict(kind='mask', description=description)
        if 'first_table_forced_background_mask' in debug:
            meta['first_surround_mask']['description'] = (
                'Union of per-region reference domains; anchored rings may include separately validated changed-color table.')
            for key, description in {
                'first_table_proposal_mask': 'Voted connected color component; this proposal alone is not trusted table.',
                'first_table_reference_domain_mask': 'Inset geometric anchor reference domain before depth validity and residual checks.',
                'first_table_reference_mask': 'Measured anchor table reference pixels passing depth checks, before per-object distance limits.',
                'first_table_forced_background_mask': 'Measured table-consistent pixels excluded from object candidates.',
                'first_table_raw_depth_foreground_mask': 'Positive raw depth change above depth_threshold_m, prior to table-offset correction.',
            }.items():
                meta[key] = dict(kind='mask', description=description)
            meta['first_table_raw_depth_difference_m'] = dict(
                kind='numeric', unit='m', description='Empty minus current depth before table-offset correction.')
            meta['depth_difference_m'] = dict(kind='numeric', unit='m',
                description='Empty minus current depth, offset-corrected when table reference is accepted.')
        for key in debug:
            if key.startswith('first_failed_'):
                meta[key] = dict(kind='overlapping_condition', description='Region gate failed: '+key[13:-5]+'. Other conditions may also fail. Unevaluated depth gates are not failures.')
        for key in ('first_mean_delta_m', 'first_median_delta_m', 'first_required_delta_m'):
            meta[key] = dict(kind='numeric', unit='m', description='Region minus surrounding table signed depth statistic, or required separation. NaN = insufficient evidence.')
        meta['first_color_group_map'] = dict(kind='label', description='Color prototype groups; matches color_group in first_statistics. One group can contain multiple disconnected regions.')
        meta['first_region_label_map'] = dict(kind='label', description='Connected-region labels; matches id in first_statistics. Zero means unassigned.')
    out['diagnostic_metadata'] = meta
    return out


def growth_boundary_diagnostics(lab, prototype, domain, selected, tolerance):
    """Inspect the final selected boundary without extending the selection.

    Distances use retained source prototypes, exactly as the growth rule does.
    This describes the next possible step, not a global reachability proof.
    """
    from ._rgbd_fusion import _shift
    scope = domain & ~selected
    adjacent = np.zeros(scope.shape, bool)
    compatible = adjacent.copy()
    source_min = np.full(scope.shape, np.inf)
    local_min = source_min.copy()
    limit = tolerance ** 2
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dy or dx):
                continue
            available = scope & _shift(selected, dy, dx, False)
            source2 = np.sum((lab - _shift(prototype, dy, dx)) ** 2, axis=2)
            local2 = np.sum((lab - _shift(lab, dy, dx)) ** 2, axis=2)
            adjacent |= available
            compatible |= available & (source2 <= limit) & (local2 <= limit)
            np.minimum(source_min, np.where(available, source2, np.inf), out=source_min)
            np.minimum(local_min, np.where(available, local2, np.inf), out=local_min)
    return dict(
        reason_growth_no_selected_neighbor_mask=scope & ~adjacent,
        reason_growth_color_failed_mask=adjacent & ~compatible,
        reason_growth_compatible_next_step_mask=compatible,
        growth_boundary_source_color_distance=np.where(adjacent, np.sqrt(source_min), np.nan),
        growth_boundary_neighbor_color_distance=np.where(adjacent, np.sqrt(local_min), np.nan),
        diagnostic_metadata={
            'reason_growth_color_failed_mask': {'description': 'No selected neighbor passes BOTH prototype and local color tolerances. Tests are paired per neighbor.'},
            'reason_growth_no_selected_neighbor_mask': {'description': 'No selected neighbor at final growth boundary; not proof of global disconnection.'},
            'reason_growth_compatible_next_step_mask': {'description': 'Could be added by one more color-compatible step; not selected within the configured growth budget.'},
            'growth_boundary_source_color_distance': {'unit': 'OpenCV Lab', 'threshold': tolerance, 'comparison': '<=', 'description': 'Minimum over selected neighbors, independently of local-distance minimum.'},
            'growth_boundary_neighbor_color_distance': {'unit': 'OpenCV Lab', 'threshold': tolerance, 'comparison': '<=', 'description': 'Minimum over selected neighbors, independently of prototype-distance minimum.'}})
