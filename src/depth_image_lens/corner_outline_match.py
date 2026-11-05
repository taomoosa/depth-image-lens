"""Interior-corner registration with bounded axial scale and edge validation.

Corners determine the transform; endpoints never participate in its fit.
Optional bounded skips apply only to current corners; all intervening geometry
remains in edge validation. This is a finite hypothesis matcher, not a garment
state or physical strain estimator.
"""
from copy import deepcopy
from itertools import combinations
import numpy as np

from .outline_match import (
    PartialOutlineMatcher, PartialOutlineMatchResult, OutlineArticulation,
    _points, _geometry, _sample, _simplified_indices, _crosses, _turn,
    _angle_difference,
)


def _rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _corners(points, tolerance, threshold, closed, spacing):
    ids = _simplified_indices(points, tolerance, closed)
    q = points[ids]
    if closed:
        left, right = q-np.roll(q, 1, axis=0), np.roll(q, -1, axis=0)-q
    else:
        left, right = q[1:-1]-q[:-2], q[2:]-q[1:-1]
        ids = ids[1:-1]
    angles = np.degrees(np.arctan2(left[:, 0]*right[:, 1]-left[:, 1]*right[:, 0],
                                   (left*right).sum(axis=1)))
    s, perimeter = _geometry(points, closed)
    selected = []
    for i in np.argsort(-np.abs(angles), kind='stable'):
        if abs(angles[i]) < threshold:
            continue
        distance = np.abs(s[ids[i]]-s[np.asarray(selected, dtype=int)])
        if closed:
            distance = np.minimum(distance, perimeter-distance)
        if np.all(distance >= spacing):
            selected.append(int(ids[i]))
    return np.asarray(sorted(selected), dtype=int)


def _fit_axes(target, current, axis_angle, min_condition):
    """Least-squares R diag(sx,sy) in declared target axes, without shear.

    Eliminating sx/sy leaves a 2x2 Rayleigh quotient in (cos(theta),sin(theta)).
    A positive interior solution is required; reflected/collapsed fits fail.
    """
    basis = _rotation(np.radians(axis_angle))
    t = (target-target.mean(axis=0)) @ basis
    q = current-current.mean(axis=0)
    singular = np.linalg.svd(t, compute_uv=False)
    condition = float(singular[-1]/singular[0]) if singular[0] > 1e-9 else 0.
    if condition < min_condition:
        return None
    denominator = (t*t).sum(axis=0)
    a = np.array([t[:, 0]@q[:, 0], t[:, 0]@q[:, 1]])
    b = np.array([t[:, 1]@q[:, 1], -t[:, 1]@q[:, 0]])
    _, vectors = np.linalg.eigh(np.outer(a, a)/denominator[0] + np.outer(b, b)/denominator[1])
    direction = vectors[:, -1]
    scales = np.array([a@direction, b@direction])/denominator
    if np.all(scales < 0):
        scales, direction = -scales, -direction
    if np.any(scales <= 1e-6):
        return None
    theta = np.arctan2(direction[1], direction[0])
    matrix = _rotation(theta) @ np.diag(scales) @ basis.T
    translation = current.mean(axis=0)-matrix@target.mean(axis=0)
    rotation = float((np.degrees(theta)-axis_angle+180) % 360-180)
    return matrix, translation, scales, rotation, condition


def _arc(points, start, end):
    s, perimeter = _geometry(points, True)
    shift = np.floor(start/perimeter)*perimeter
    vertices = np.r_[s[:-1]+shift, s[:-1]+shift+perimeter]
    positions = np.r_[start, vertices[(vertices > start+1e-9) & (vertices < end-1e-9)], end]
    return _sample(points, positions, True)


def _project(points, target, start, end):
    arc = _arc(points, start, end)
    s, _ = _geometry(arc, False)
    delta = np.diff(arc, axis=0)
    length2 = (delta*delta).sum(axis=1)
    u = np.clip(((target-arc[:-1])*delta).sum(axis=1)/np.maximum(length2, 1e-20), 0, 1)
    closest = arc[:-1]+u[:, None]*delta
    best = int(np.argmin(np.linalg.norm(closest-target, axis=1)))
    return float(start+s[best]+u[best]*np.sqrt(length2[best]))


def _supported(points, observed, start, end):
    s, perimeter = _geometry(points, True)
    shift = np.floor(start/perimeter)*perimeter
    lo = np.r_[s[:-1]+shift, s[:-1]+shift+perimeter]
    hi = np.r_[s[1:]+shift, s[1:]+shift+perimeter]
    crossed = (lo < end-1e-9) & (hi > start+1e-9)
    return bool(np.tile(observed, 2)[crossed].all())


def _range_score(value, limits):
    # The neutral value 1 scores zero; a bound scores 1, outside scores >1.
    return float(abs(np.log(value))/abs(np.log(limits[0] if value < 1 else limits[1])))


def _correspondence_plans(target_count, current_count, max_skipped):
    """Ordered interior choices and endpoint windows, with a shared skip budget.

    The first matched corner is fixed by the caller. A tail window may extend
    past additional current corners; actual skips are counted after projection.
    Never reuse a current corner within a correspondence, even across the seam.
    """
    budget = min(max_skipped, current_count-target_count)
    plans = []
    for internal in range(budget+1):
        last = target_count-1+internal
        for middle in combinations(range(1, last), target_count-2):
            offsets = np.array((0, *middle, last), dtype=int)
            tails = [(before, after) for before in range(budget-internal+1)
                     for after in range(budget-internal-before+1)]
            plans.append((offsets, tails))
    return plans


class CornerOutlineMatcher(PartialOutlineMatcher):
    """Fit internal bends, then check every intervening edge and both tails.

    Inherits match/match_contour; usable as outline.match_partial(matcher=...).
    Target endpoints are excluded from corner detection AND transformation fit.
    Axes are fixed in the supplied target coordinates, rotated by axis_angle_deg.
    At least three non-collinear interior corners are needed. Auto-detected bends
    use RDP plus an angle threshold, independently of OuterContourResult.corners.
    """
    def __init__(self, *, corner_tolerance_px=1.5, corner_turn_deg=20., corner_spacing_px=5.,
                 axis_angle_deg=0., scale_range=(.5, 2.), axis_ratio_range=(.75, 4/3),
                 max_rotation_deg=180., max_corner_rms=.025, max_corner_error=.06,
                 max_edge_rms=.06, max_edge_error=.15, max_edge_length_error=.20,
                 max_edge_turn_deg=25., max_edge_excess_turn_deg=60.,
                 max_edge_unresolved_ratio=.06, min_corner_condition=.08,
                 fraction_range=(.15, .65), min_arc_length_px=30.,
                 edge_samples=32, simplify_px=.75, max_candidates=5,
                 max_hypotheses=4096, ambiguity_margin=.12,
                 ambiguity_rotation_deg=15., ambiguity_scale_ratio=1.15,
                 max_skipped_corners=0):
        values = locals().copy()
        integer_limits = dict(edge_samples=(16, 128), max_candidates=(2, 20), max_hypotheses=(16, 4096),
                              max_skipped_corners=(0, 2))
        for name, (lo, hi) in integer_limits.items():
            if type(values[name]) is not int or not lo <= values[name] <= hi:
                raise ValueError(f'{name} must be integer {lo}..{hi}')
        ranges = ('scale_range', 'axis_ratio_range', 'fraction_range')
        for name in ranges:
            v = np.asarray(values[name])
            if v.shape != (2,) or v.dtype.kind not in 'iuf' or not np.isfinite(v).all():
                raise ValueError(f'{name} must contain two finite numbers')
            if name == 'fraction_range':
                valid = 0 < v[0] < v[1] < 1
            else:
                valid = 0 < v[0] < 1 < v[1]
            if not valid:
                raise ValueError(f'invalid {name}')
        for name, v in values.items():
            if name in ('self', '__class__') or name in integer_limits or name in ranges:
                continue
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)) or not np.isfinite(v):
                raise ValueError(f'{name} must be finite numeric')
            if name not in ('axis_angle_deg', 'simplify_px') and v <= 0:
                raise ValueError(f'{name} must be positive')
        if not (0 <= simplify_px <= 5 and 0 < corner_turn_deg < 180 and
                0 < min_corner_condition < 1 and 0 < max_rotation_deg <= 180 and
                0 < ambiguity_rotation_deg <= 180 and abs(axis_angle_deg) <= 180 and
                ambiguity_scale_ratio > 1):
            raise ValueError('invalid simplification, angle, condition or ambiguity limit')
        self.parameters = {k: (list(map(float, v)) if k in ranges else
                               int(v) if k in integer_limits else float(v))
                           for k, v in values.items() if k not in ('self', '__class__')}

    def _match(self, target, records, articulation):
        p = self.parameters
        target = _points(target, 'target_points_xy')
        if len(target) > 2048 or _crosses(target):
            raise ValueError('target must be a simple open polyline of at most 2048 vertices')
        if articulation is not None and not isinstance(articulation, OutlineArticulation):
            raise ValueError('articulation must be an OutlineArticulation or None')
        variants = [(0., target)] if articulation is None else articulation.variants(target)
        if not variants:
            raise ValueError('no non-self-intersecting template variants')
        ids = _corners(target, p['corner_tolerance_px'], p['corner_turn_deg'], False, p['corner_spacing_px'])
        reasons, sources, candidates = [], [], []
        incomplete, evaluated = not records, 0
        pruned, edge_evaluations, best_score = 0, 0, float('inf')
        budget_limited = False
        if len(ids) < 3:
            reasons.append('at_least_three_interior_corners_required')
        elif len(ids) > 16:
            reasons.append('target_corner_budget_exceeded')
        elif _fit_axes(target[ids], target[ids], p['axis_angle_deg'], p['min_corner_condition']) is None:
            reasons.append('interior_corners_are_nearly_collinear')
        if not reasons:
            for cid, points, observed, reason in records:
                source = dict(contour_id=int(cid), reason=reason)
                sources.append(source)
                if points is None or reason is not None:
                    incomplete = True
                    continue
                incomplete |= not bool(observed.all())
                if len(points) > 4096:
                    source['reason'] = 'corner_topology_budget_exceeded'
                    budget_limited = True
                    incomplete = True
                    continue
                for reverse in (False, True):
                    q = points[::-1].copy() if reverse else points
                    good = np.roll(observed[::-1], -1) if reverse else observed
                    corners = _corners(q, p['corner_tolerance_px'], p['corner_turn_deg'], True, p['corner_spacing_px'])
                    source['detected_corner_count'] = len(corners)
                    if len(corners) < len(ids):
                        source['reason'] = 'insufficient_current_corners'
                        incomplete = True
                        continue
                    plans = _correspondence_plans(len(ids), len(corners), p['max_skipped_corners'])
                    required = len(corners)*len(variants)*sum(len(tails) for _, tails in plans)
                    if len(corners) > 64 or evaluated+required > p['max_hypotheses']:
                        source['reason'] = 'hypothesis_budget_exceeded'
                        budget_limited = True
                        incomplete = True
                        continue
                    s, perimeter = _geometry(q, True)
                    if perimeter*p['fraction_range'][1] < p['min_arc_length_px']:
                        source['reason'] = 'insufficient_arc_length'
                        incomplete = True
                        continue
                    positions = s[corners]
                    for k in range(len(corners)):
                        for offsets, tails in plans:
                            order = k+offsets
                            current_ids = corners[order % len(corners)]
                            interior = positions[order % len(corners)]+(order//len(corners))*perimeter
                            for angle, variant in variants:
                                evaluated += len(tails)
                                fit = _fit_axes(variant[ids], q[current_ids], p['axis_angle_deg'], p['min_corner_condition'])
                                if fit is None:
                                    continue
                                angle_limit = articulation.max_angle_deg if articulation is not None else 0.
                                quality = self._corner_quality(variant, ids, q, interior, fit, angle, angle_limit)
                                if quality is None:
                                    continue
                                # Safe lower bound: edge errors can only raise the
                                # score. Keep EVERY potential passing hypothesis
                                # for ambiguity, including when a perfect fit exists.
                                if p['max_skipped_corners'] and max(quality[2].values()) > max(1., best_score):
                                    pruned += len(tails)
                                    continue
                                for before, after in tails:
                                    left, right = k-1-before, order[-1]+1+after
                                    prev = positions[left % len(corners)]+(left//len(corners))*perimeter
                                    nxt = positions[right % len(corners)]+(right//len(corners))*perimeter
                                    row = self._candidate(variant, ids, q, good, interior, prev, nxt, fit, quality, angle, cid, reverse)
                                    edge_evaluations += 1
                                    if row is None:
                                        continue
                                    # Geometry is uncut: report unmatched corners
                                    # actually inside the projected interval, including tails.
                                    extra = np.setdiff1d(np.arange(left+1, right), order)
                                    locations = positions[extra % len(corners)]+(extra//len(corners))*perimeter
                                    extra_ids = corners[extra[(locations > row['_bounds'][0]+1e-9) &
                                                              (locations < row['_bounds'][1]-1e-9)] % len(corners)]
                                    row.pop('_bounds')
                                    row['skipped_corner_indices'] = (len(q)-1-extra_ids if reverse else extra_ids).tolist()
                                    row['skipped_corner_xy'] = q[extra_ids].tolist()
                                    row['skipped_corner_count'] = len(extra_ids)
                                    row['observed_corner_indices'] = (len(q)-1-current_ids if reverse else current_ids).tolist()
                                    best_score = min(best_score, row['score'])
                                    candidates.append(row)
        candidates.sort(key=lambda r: (not r['passed'], r['score'], r['skipped_corner_count'],
                                       abs(r['articulation_angle_deg'])))
        best = candidates[0] if candidates else None
        passing = [r for r in candidates if r['passed']]
        ambiguity, alternatives = [], []
        if passing:
            best = passing[0]
            for row in passing[1:]:
                # A smaller deformation prior does not make an unidentifiable
                # physical explanation identifiable from the observed shape.
                if row['geometry_score'] > best['geometry_score']+p['ambiguity_margin']:
                    continue
                if not self._same_location(best, row):
                    ambiguity.append('multiple_distinct_visible_arcs_fit')
                elif (abs(_angle_difference(np.radians(row['rotation_deg']), np.radians(best['rotation_deg']))) >= np.radians(p['ambiguity_rotation_deg']) or
                      abs(row['articulation_angle_deg']-best['articulation_angle_deg']) >= p['ambiguity_rotation_deg'] or
                      max(row['scale_x']/best['scale_x'], best['scale_x']/row['scale_x'],
                          row['scale_y']/best['scale_y'], best['scale_y']/row['scale_y']) >= p['ambiguity_scale_ratio']):
                    ambiguity.append('multiple_transform_explanations_fit')
                    alternatives.append({k: row[k] for k in ('score', 'geometry_score', 'scale_x', 'scale_y',
                                                             'rotation_deg', 'articulation_angle_deg')})
        if reasons:
            status, matched = 'unknown', None
        elif budget_limited:
            status, matched, reasons = 'unknown', None, ['search_budget_exceeded']
        elif ambiguity:
            status, matched, reasons = 'ambiguous', None, sorted(set(ambiguity))
        elif passing:
            status, matched = 'matched', True
        elif incomplete or any(r['score'] <= 1 and not r['supported'] for r in candidates):
            status, matched, reasons = 'unknown', None, ['incomplete_or_uncertain_corner_evidence']
        else:
            status, matched, reasons = 'not_matched', False, ['no_corner_correspondence_passed_limits']
        # Include distinct locations before filling remaining slots with transform alternatives.
        display = []
        for row in candidates:
            if not any(self._same_location(row, other) for other in display):
                display.append(row)
        display = display[:p['max_candidates']]
        for row in candidates:
            if len(display) >= p['max_candidates']:
                break
            if not any(row is r for r in display):
                display.append(row)
        for row in display:
            row.pop('_interval')
            row['points_xy'] = row['points_xy'].tolist()
            row['fitted_target_xy'] = row['fitted_target_xy'].tolist()
        config = deepcopy(p)
        config['articulation'] = None if articulation is None else dict(
            start_index=articulation.start_index, end_index=articulation.end_index,
            pivot_xy=np.asarray(articulation.pivot_xy, float).tolist(),
            max_angle_deg=float(articulation.max_angle_deg), angle_step_deg=float(articulation.angle_step_deg))
        diagnostics = dict(reasons=reasons, source_contours=sources, evaluated_hypotheses=evaluated,
            corner_bound_pruned_hypotheses=pruned, endpoint_window_evaluations=edge_evaluations,
            search=('bounded_current_corner_skips_and_projected_endpoints' if p['max_skipped_corners'] else
                    'consecutive_internal_corner_correspondences_and_projected_endpoints'), search_complete=False,
            target_corner_indices=ids.tolist(), target_corner_count=len(ids), candidate_found=bool(passing),
            alternative_transforms=alternatives[:p['max_candidates']],
            scope='visible_partial_outline_not_whole_object_state', score_kind='maximum_limit_ratio_not_probability',
            endpoint_role='excluded_from_transform_fit_included_in_edge_validation',
            transform_order='articulation_then_target_axis_scaling_then_global_rotation_then_translation')
        return PartialOutlineMatchResult(status, matched, best, display, config, diagnostics)

    @staticmethod
    def _same_location(a, b):
        if a['contour_id'] != b['contour_id']:
            return False
        # Compare the same physical interval, allowing reversed traversal.
        x, y = a['_interval'], b['_interval']
        overlap = max(max(0., min(x[1], y[1]+shift)-max(x[0], y[0]+shift)) for shift in (-1, 0, 1))
        return overlap/max(1e-12, min(x[1]-x[0], y[1]-y[0])) >= .6

    def _corner_quality(self, target, ids, current, interior, fit, angle, angle_limit):
        p = self.parameters
        matrix, translation, scales, rotation, condition = fit
        fitted = target@matrix.T+translation
        residual = np.linalg.norm(fitted[ids]-_sample(current, interior, True), axis=1)
        span = np.linalg.norm(np.ptp(fitted[ids], axis=0))
        if span < 1e-6:
            return None
        metrics = dict(corner_rms=float(np.sqrt(np.mean(residual**2))/span),
                       corner_error=float(residual.max()/span))
        limits = dict(corner_rms=p['max_corner_rms'], corner_error=p['max_corner_error'])
        ratios = {k: metrics[k]/v for k, v in limits.items()}
        ratios.update(scale_x=_range_score(scales[0], p['scale_range']),
                      scale_y=_range_score(scales[1], p['scale_range']),
                      axis_ratio=_range_score(scales[0]/scales[1], p['axis_ratio_range']),
                      rotation=abs(rotation)/p['max_rotation_deg'],
                      articulation=abs(angle)/angle_limit if angle_limit else 0.)
        return fitted, metrics, ratios

    def _candidate(self, target, ids, current, observed, interior, prev, nxt, fit, quality, angle, cid, reverse):
        p = self.parameters
        matrix, translation, scales, rotation, condition = fit
        fitted, metrics, initial_ratios = quality
        ratios = initial_ratios.copy()
        start = _project(current, fitted[0], prev, interior[0])
        end = _project(current, fitted[-1], interior[-1], nxt)
        _, perimeter = _geometry(current, True)
        length = end-start
        if not p['fraction_range'][0] <= length/perimeter <= p['fraction_range'][1]:
            return None
        boundaries = np.r_[start, interior, end]
        target_ids = np.r_[0, ids, len(target)-1]
        edges, samples, templates = [], [], []
        for edge, (a, b) in enumerate(zip(target_ids[:-1], target_ids[1:])):
            t = fitted[a:b+1]
            q = _arc(current, boundaries[edge], boundaries[edge+1])
            if _geometry(q, False)[1] < 1e-6:
                return None
            t = t[_simplified_indices(t, p['simplify_px'])]
            q = q[_simplified_indices(q, p['simplify_px'])]
            _, tl = _geometry(t, False)
            _, ql = _geometry(q, False)
            unit = np.linspace(0, 1, p['edge_samples'])
            ts, qs = _sample(t, unit*tl, False), _sample(q, unit*ql, False)
            distances = np.linalg.norm(ts-qs, axis=1)
            tt, qt = _turn(ts, 2), _turn(qs, 2)
            unresolved = max(0., (1-np.linalg.norm(np.diff(qs, axis=0), axis=1).sum()/ql)-
                             (1-np.linalg.norm(np.diff(ts, axis=0), axis=1).sum()/tl))
            em = dict(rms=float(np.sqrt(np.mean(distances**2))/tl), error=float(distances.max()/tl),
                      length_error=float(abs(ql/tl-1)),
                      turn_deg=float(np.degrees(np.sqrt(np.mean(_angle_difference(qt, tt)**2)))),
                      excess_turn_deg=float(max(0., np.degrees((np.abs(qt).sum()-np.abs(tt).sum())/2))),
                      unresolved_ratio=float(unresolved))
            er = {name: value/p['max_edge_'+name] for name, value in em.items()}
            edge_score = max(er.values())
            edges.append(dict(index=edge, target_index_range=[int(a), int(b)],
                role='endpoint_tail' if edge in (0, len(target_ids)-2) else 'between_corners',
                start_xy=qs[0].tolist(), end_xy=qs[-1].tolist(),
                target_length_px=float(tl), observed_length_px=float(ql), length_ratio=float(ql/tl),
                rms_px=float(np.sqrt(np.mean(distances**2))), metrics=em,
                score=float(edge_score), passed=bool(edge_score <= 1)))
            samples.append(qs); templates.append(ts)
        ratios['edges'] = max(e['score'] for e in edges)
        score = float(max(ratios.values()))
        supported = _supported(current, observed, start, end) and length >= p['min_arc_length_px']
        # Reversed input starts at the last original vertex.
        seam = perimeter-np.linalg.norm(current[-1]-current[0]) if reverse else 0.
        low = (seam-end if reverse else start) % perimeter
        interval = [float(low/perimeter), float((low+length)/perimeter)]
        return dict(contour_id=int(cid), score=score,
            geometry_score=float(max(ratios[k] for k in ('corner_rms', 'corner_error', 'edges'))),
            passed=bool(score <= 1 and supported),
            supported=bool(supported), reasons=([k+'_exceeds_limit' for k,v in ratios.items() if v > 1] +
                ([] if supported else ['unobserved_edges_or_insufficient_support'])),
            metrics=metrics, limit_ratios=ratios, edges=edges, fraction=float(length/perimeter),
            arc_length_px=float(length), start_xy=samples[0][0].tolist(), end_xy=samples[-1][-1].tolist(),
            points_xy=np.vstack(samples), fitted_target_xy=np.vstack(templates), reversed=reverse,
            articulation_angle_deg=float(angle), scale=float(np.sqrt(scales.prod())),
            scale_x=float(scales[0]), scale_y=float(scales[1]), axis_ratio=float(scales[0]/scales[1]),
            axis_angle_deg=p['axis_angle_deg'], rotation_deg=rotation,
            translation_xy=translation.tolist(), matrix_xy=matrix.tolist(),
            corner_condition=condition, target_corner_indices=ids.tolist(),
            target_corner_xy=target[ids].tolist(), fitted_corner_xy=fitted[ids].tolist(),
            observed_corner_xy=_sample(current, interior, True).tolist(),
            _interval=interval, _bounds=(start, end))
