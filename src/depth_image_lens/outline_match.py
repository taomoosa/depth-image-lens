"""Constrained open-to-closed outline matching; scores are not probabilities.

Only contiguous, fully observed arcs can support a match. No DTW, reflection,
occlusion completion, or unconstrained nonrigid registration is performed.
"""
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from .debug import DebugResultMixin, _opencv


def _points(value, name):
    a = np.asarray(value)
    if a.dtype.kind not in 'iuf' or a.ndim != 2 or a.shape[1] != 2 or len(a) < 3:
        raise ValueError(f'{name} must be a numeric Nx2 array with at least 3 points')
    a = a.astype(float, copy=True)
    if not np.isfinite(a).all() or np.max(np.abs(a)) > 1e9:
        raise ValueError(f'{name} must contain finite coordinates within +/-1e9')
    if np.any(np.linalg.norm(np.diff(a, axis=0), axis=1) <= 1e-9):
        raise ValueError(f'{name} must not contain consecutive duplicate points')
    if np.linalg.norm(a[-1] - a[0]) <= 1e-9:
        raise ValueError(f'{name} must not repeat its first point at the end')
    if len(a) > 20000:
        raise ValueError(f'{name} exceeds the 20000 point input budget')
    return a


def _geometry(a, closed):
    steps = np.linalg.norm(np.diff(np.vstack([a, a[:1]]) if closed else a, axis=0), axis=1)
    return np.r_[0., np.cumsum(steps)], float(steps.sum())


def _sample(a, positions, closed):
    s, length = _geometry(a, closed)
    if closed:
        a = np.vstack([a, a[:1]])
        positions = np.mod(positions, length)
    return np.stack([np.interp(positions, s, a[:, d]) for d in range(2)], axis=-1)


def _turn(a, radius):
    left = a[..., radius:-radius, :] - a[..., :-2*radius, :]
    right = a[..., 2*radius:, :] - a[..., radius:-radius, :]
    cross = left[..., 0]*right[..., 1] - left[..., 1]*right[..., 0]
    dot = (left*right).sum(axis=-1)
    return np.arctan2(cross, dot)


def _angle_difference(a, b):
    return np.arctan2(np.sin(a-b), np.cos(a-b))


def _simplified_indices(points, tolerance, closed=False):
    """Ramer-Douglas-Peucker with a distance-to-segment error in pixels.

    Keep indices for conservative observation provenance. Closed curves are
    split at a farthest vertex so the arbitrary cyclic seam is not a zero chord.
    """
    if tolerance == 0 or len(points) <= 3:
        return np.arange(len(points))
    n = len(points)
    p = np.vstack([points, points[:1]]) if closed else points
    split = int(np.argmax(np.linalg.norm(points-points[0], axis=1))) if closed else n-1
    stack = [(0, split), (split, n)] if closed else [(0, n-1)]
    keep = {0, split, n if closed else n-1}
    while stack:
        a, b = stack.pop()
        if b <= a+1:
            continue
        v = p[b]-p[a]
        length2 = float(v@v)
        u = np.zeros(b-a-1) if length2 == 0 else np.clip((p[a+1:b]-p[a])@v/length2, 0, 1)
        distances = np.linalg.norm(p[a+1:b]-(p[a]+u[:,None]*v), axis=1)
        offset = int(np.argmax(distances))
        if distances[offset] > tolerance:
            i = a+1+offset
            keep.add(i)
            stack.extend([(a,i),(i,b)])
    ids = np.array(sorted(i for i in keep if i < n), dtype=int)
    if closed and len(ids) < 3:
        return np.arange(n)
    # Do not introduce a crossing when simplifying close portions of a boundary.
    # Above the topology budget, preserve the supplied geometry instead.
    if len(ids) > 4096 or _crosses(points[ids], closed):
        return np.arange(n)
    return ids


def _crosses(a, closed=False):
    """Proper crossings or nonadjacent contacts, for sparse template polygons."""
    b = np.vstack([a, a[:1]]) if closed else a
    starts, ends = b[:-1], b[1:]
    edges = ends-starts
    next_edges = np.roll(edges, -1, axis=0) if closed else edges[1:]
    previous = edges if closed else edges[:-1]
    cross_adjacent = previous[:, 0]*next_edges[:, 1]-previous[:, 1]*next_edges[:, 0]
    if np.any((np.abs(cross_adjacent) <= 1e-12) & ((previous*next_edges).sum(axis=1) < 0)):
        return True
    for i in range(len(starts)-2):
        js = np.arange(i+2, len(starts))
        if closed and i == 0:
            js = js[js != len(starts)-1]
        if not len(js):
            continue
        c, d = starts[js], ends[js]
        u, v = starts[i], ends[i]
        overlap = (np.maximum(np.minimum(c, d), np.minimum(u, v)) <=
                   np.minimum(np.maximum(c, d), np.maximum(u, v)) + 1e-9).all(axis=1)
        def cross(x, y):
            return x[..., 0]*y[..., 1] - x[..., 1]*y[..., 0]
        intersects = ((cross(v-u, c-u)*cross(v-u, d-u) <= 0) &
                      (cross(d-c, u-c)*cross(d-c, v-c) <= 0))
        if np.any(overlap & intersects):
            return True
    return False


@dataclass(frozen=True)
class OutlineArticulation:
    """Rotate one contiguous group of *template vertices* around a known pivot.

    start_index/end_index are inclusive, before resampling. Adjacent fixed points
    reconnect by straight segments. This is a bounded silhouette hypothesis, not
    cloth physics. Use a sparse anatomical polyline, not arbitrary pixel indices.
    """
    start_index: int
    end_index: int
    pivot_xy: tuple
    max_angle_deg: float = 30.
    angle_step_deg: float = 10.

    def variants(self, points):
        a = _points(points, 'target_points_xy')
        if any(type(i) is not int for i in (self.start_index, self.end_index)):
            raise ValueError('articulation indices must be integers')
        if not 0 <= self.start_index <= self.end_index < len(a):
            raise ValueError('articulation indices are outside the template')
        if self.start_index == 0 and self.end_index == len(a)-1:
            raise ValueError('articulation must leave at least one template point fixed')
        pivot = np.asarray(self.pivot_xy)
        if (pivot.shape != (2,) or pivot.dtype.kind not in 'iuf' or
                not np.isfinite(pivot).all() or np.max(np.abs(pivot)) > 1e9):
            raise ValueError('pivot_xy must contain two finite numbers within +/-1e9')
        for v in (self.max_angle_deg, self.angle_step_deg):
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)) or not np.isfinite(v):
                raise ValueError('articulation angles must be finite numbers')
        if not 0 <= self.max_angle_deg <= 45 or not 1 <= self.angle_step_deg <= 45:
            raise ValueError('max_angle_deg must be 0..45 and angle_step_deg 1..45')
        count = int(np.ceil(self.max_angle_deg/self.angle_step_deg))
        angles = np.linspace(-self.max_angle_deg, self.max_angle_deg, 2*count+1)
        output = []
        for angle in angles:
            rad = np.radians(angle)
            rotation = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            q = a.copy()
            sl = slice(self.start_index, self.end_index+1)
            q[sl] = (q[sl]-pivot) @ rotation.T + pivot
            if np.any(np.linalg.norm(np.diff(q, axis=0), axis=1) <= 1e-9) or _crosses(q):
                continue
            output.append((float(angle), q))
        return output


@dataclass(frozen=True)
class PartialOutlineMatchResult(DebugResultMixin):
    status: str
    matched: object
    best: object
    candidates: list
    parameters: dict
    diagnostics: dict

    def to_dict(self):
        return deepcopy(dict(status=self.status, matched=self.matched, best=self.best,
                             candidates=self.candidates, parameters=self.parameters,
                             diagnostics=self.diagnostics))

    def render_debug(self, color, *, color_order='RGB', candidate_index=0):
        """Overlay observed arc (cyan), fitted template (orange), and endpoints.

        Supply the current image, in the same XY grid used for matching. The
        explicit save_debug method is inherited; no matching call writes files.
        """
        cv = _opencv()
        a = np.asarray(color)
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 2:
            raise ValueError('color must be uint8 HxWx3')
        if color_order not in ('RGB', 'BGR'):
            raise ValueError('color_order must be RGB or BGR')
        if type(candidate_index) is not int or candidate_index < 0 or candidate_index >= max(1, len(self.candidates)):
            raise ValueError('candidate_index must select an available candidate')
        canvas = (a if color_order == 'RGB' else a[..., ::-1]).copy()
        row = self.candidates[candidate_index] if self.candidates else None
        if row is not None:
            for key, ink, width in [('points_xy', (40, 210, 240), 2),
                                    ('fitted_target_xy', (255, 150, 45), 1)]:
                points = np.rint(row[key]).clip(-2**30, 2**30).astype(np.int32)
                cv.polylines(canvas, [points], False, ink, width, cv.LINE_AA)
            for xy in (row['start_xy'], row['end_xy']):
                point = tuple(np.rint(xy).clip(-2**30, 2**30).astype(int))
                cv.circle(canvas, point, 4, (250, 250, 250), 1, cv.LINE_AA)
        label = self.status + ('' if row is None else f"  score={row['score']:.2f}")
        cv.putText(canvas, label, (5, 18), cv.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv.LINE_AA)
        if row is not None:
            detail = f"angle={row['articulation_angle_deg']:.0f} deg  observed={row['supported']}"
            cv.putText(canvas, detail, (5, 34), cv.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1, cv.LINE_AA)
        return canvas if color_order == 'RGB' else canvas[..., ::-1].copy()


class PartialOutlineMatcher:
    """Finite coarse-to-fine arc search with a bounded template family.

    ``match`` consumes OuterContourResult; ``match_contour`` consumes a cyclic
    ordered polyline without a duplicated closing point. XY coordinates, no writes.
    A match concerns only the supplied visible exterior, never the whole garment.
    """
    def __init__(self, *, fraction_range=(.20, .55), search_samples=128,
                 length_samples=25, sample_count=64, max_rms=.020,
                 max_point_error=.055, max_fine_turn_deg=20.,
                 max_coarse_turn_deg=25., max_excess_turn_deg=60.,
                 max_unresolved_arc_ratio=.06, ambiguity_margin=.12,
                 min_arc_length_px=30., max_candidates=5, simplify_px=.75):
        limits = dict(search_samples=(32, 512), length_samples=(3, 101),
                      sample_count=(32, 128), max_candidates=(2, 20))
        values = locals().copy()
        for name, (lo, hi) in limits.items():
            if type(values[name]) is not int or not lo <= values[name] <= hi:
                raise ValueError(f'{name} must be integer {lo}..{hi}')
        f = np.asarray(fraction_range)
        if f.shape != (2,) or f.dtype.kind not in 'iuf' or not np.isfinite(f).all() or not 0 < f[0] < f[1] < 1:
            raise ValueError('fraction_range must satisfy 0 < minimum < maximum < 1')
        numeric = ('max_rms', 'max_point_error', 'max_fine_turn_deg', 'max_coarse_turn_deg',
                   'max_excess_turn_deg', 'max_unresolved_arc_ratio', 'ambiguity_margin', 'min_arc_length_px')
        for name in numeric:
            v = values[name]
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)) or not np.isfinite(v) or v <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if max_fine_turn_deg > 180 or max_coarse_turn_deg > 180:
            raise ValueError('turn RMS limits must be <=180 degrees')
        if (isinstance(simplify_px, (bool, np.bool_)) or not isinstance(simplify_px, (int, float, np.number))
                or not np.isfinite(simplify_px) or not 0 <= simplify_px <= 5):
            raise ValueError('simplify_px must be finite in 0..5 processing pixels')
        self.parameters = {name: int(values[name]) for name in limits}
        self.parameters.update({name: float(values[name]) for name in numeric})
        self.parameters['fraction_range'] = list(map(float, f))
        self.parameters['simplify_px'] = float(simplify_px)

    def match(self, target_points_xy, outline, *, articulation=None):
        from .outline import OuterContourResult
        if not isinstance(outline, OuterContourResult):
            raise ValueError('outline must be an OuterContourResult; use match_contour for points')
        records = []
        ids = outline._id_map(outline.contours, 'contours')
        for cid, contour in ids.items():
            points = contour['points_xy']
            if len(points) < 3:
                records.append((cid, None, None, 'insufficient_contour'))
                continue
            edges = np.zeros(len(points), bool)
            for segment in contour['segments']:
                if segment['visibility'] == 'observed':
                    edges[(segment['start_index'] + np.arange(segment['edge_count'])) % len(edges)] = True
            reason = None if contour['status'] == 'computed' else contour['status']
            records.append((cid, _points(points, 'current contour'), edges, reason))
        return self._match(target_points_xy, records, articulation)

    def match_contour(self, target_points_xy, contour_points_xy, *, observed_edges=None, articulation=None):
        q = _points(contour_points_xy, 'contour_points_xy')
        edges = np.ones(len(q), bool) if observed_edges is None else np.asarray(observed_edges)
        if edges.dtype != bool or edges.shape != (len(q),):
            raise ValueError('observed_edges must be bool with one entry per cyclic edge')
        # The mask extractor already checks topology. Raw sparse polygons need a
        # separate check. Bound its quadratic work instead of silently skipping it.
        if len(q) > 4096:
            reason = 'raw_topology_check_budget_exceeded'
        else:
            reason = 'self_intersecting_or_touching_contour' if _crosses(q, True) else None
        return self._match(target_points_xy, [(0, q, edges.copy(), reason)], articulation)

    def _match(self, target, records, articulation):
        p = self.parameters
        t = _points(target, 'target_points_xy')
        if len(t) > 2048:
            raise ValueError('use at most 2048 target vertices')
        if _crosses(t):
            raise ValueError('target must be a simple open polyline')
        if articulation is not None and not isinstance(articulation, OutlineArticulation):
            raise ValueError('articulation must be an OutlineArticulation or None')
        variants = [(0., t)] if articulation is None else articulation.variants(t)
        if not variants:
            raise ValueError('no non-self-intersecting template variants')
        n = p['sample_count']
        unit = np.linspace(0, 1, n)
        templates = []
        for angle, vertices in variants:
            vertices = vertices[_simplified_indices(vertices, p['simplify_px'])]
            _, length = _geometry(vertices, False)
            q = _sample(vertices, unit*length, False)
            templates.append((angle, q, length))
        base_vertices = t[_simplified_indices(t, p['simplify_px'])]
        base = _sample(base_vertices, unit*_geometry(base_vertices, False)[1], False)
        # A near-straight fragment has no distinctive shape after free scaling.
        informative = np.sum(np.abs(_turn(base, 2))) >= np.radians(45)*2
        candidates = []
        evaluations = 0
        incomplete = not records
        sources = []
        for cid, contour, observed, reason in records:
            sources.append(dict(contour_id=int(cid), reason=reason))
            if contour is None or reason is not None:
                incomplete = True
                continue
            incomplete |= not bool(observed.all())
            original_length = _geometry(contour, True)[1]
            indices = _simplified_indices(contour, p['simplify_px'], True)
            missing_prefix = np.r_[0, np.cumsum(~observed)]
            ends = np.r_[indices[1:], len(contour)]
            observed = missing_prefix[ends]-missing_prefix[indices] == 0
            sources[-1].update(input_vertex_count=len(contour), matching_vertex_count=len(indices),
                               input_perimeter_px=original_length)
            contour = contour[indices]
            _, perimeter = _geometry(contour, True)
            sources[-1]['matching_perimeter_px'] = perimeter
            if perimeter*p['fraction_range'][1] < p['min_arc_length_px']:
                incomplete = True
                sources[-1]['reason'] = 'insufficient_arc_length'
                continue
            rows, count = self._search(contour, observed, templates, cid)
            evaluations += count
            if len(templates) > 1:
                # Joint spatial suppression can replace a rigid seed with a
                # rotated hypothesis whose local refinement converges worse.
                # Preserve the same zero-articulation search as articulation=None.
                rigid = [template for template in templates if template[0] == 0.]
                baseline, count = self._search(contour, observed, rigid, cid)
                evaluations += count
                rows.extend(baseline)
            candidates.extend(self._distinct(rows, p['max_candidates']+1))
        # Prefer a fully supported passing candidate to an occluded lookalike.
        candidates.sort(key=lambda r: (not (r['supported'] and r['score'] <= 1), r['score'], r['contour_id']))
        passing = [r for r in candidates if r['supported'] and r['score'] <= 1]
        best = (passing or candidates or [None])[0]
        ambiguous = len(passing) > 1 and passing[1]['score'] <= passing[0]['score'] + p['ambiguity_margin']
        if not informative:
            status, matched = 'unknown', None
            reasons = ['target_has_insufficient_turn_structure']
        elif ambiguous:
            status, matched = 'ambiguous', None
            reasons = ['multiple_distinct_visible_arcs_fit']
        elif passing:
            status, matched, reasons = 'matched', True, []
        elif incomplete or any(not r['supported'] and r['score'] <= 1 for r in candidates):
            status, matched = 'unknown', None
            reasons = ['incomplete_or_uncertain_contour_evidence']
        else:
            status, matched = 'not_matched', False
            reasons = ['no_candidate_passed_limits_in_sampled_search']
        for row in candidates:
            row.pop('_start', None)
        config = deepcopy(p)
        config['articulation'] = None if articulation is None else dict(
            start_index=articulation.start_index, end_index=articulation.end_index,
            pivot_xy=np.asarray(articulation.pivot_xy, float).tolist(),
            max_angle_deg=float(articulation.max_angle_deg), angle_step_deg=float(articulation.angle_step_deg))
        diagnostics = dict(reasons=reasons, evaluated_hypotheses=evaluations,
                           template_angles_deg=[v[0] for v in variants],
                           rigid_baseline_preserved=True,
                           search='finite_grid_with_three_local_refinements', search_complete=False,
                           source_contours=sources, candidate_found=bool(passing),
                           scope='visible_partial_outline_not_whole_object_state',
                           arc_coordinate_system='simplified_contour_arclength',
                           score_kind='maximum_limit_ratio_not_probability')
        return PartialOutlineMatchResult(status, matched, best, candidates[:p['max_candidates']], config, diagnostics)

    def _search(self, contour, observed, templates, cid):
        p = self.parameters
        starts = np.arange(p['search_samples'])/p['search_samples']
        spans = np.linspace(*p['fraction_range'], p['length_samples'])
        grid = np.array(np.meshgrid(starts, spans, indexing='ij')).reshape(2, -1).T
        rows, evaluations = self._evaluate(grid, contour, observed, templates, cid)
        # Keep independent locations, including unsupported geometric lookalikes.
        seeds = self._distinct(rows, 12)
        all_rows = seeds[:]
        ds = 1/p['search_samples']
        dl = (p['fraction_range'][1]-p['fraction_range'][0])/(p['length_samples']-1)
        for _ in range(3):
            offsets = np.array(np.meshgrid([-ds/2, 0, ds/2], [-dl/2, 0, dl/2])).reshape(2, -1).T
            refined = np.array([[r['_start'], r['fraction']] for r in seeds])[:, None, :] + offsets
            refined = refined.reshape(-1, 2)
            refined[:, 0] %= 1
            refined[:, 1] = np.clip(refined[:, 1], *p['fraction_range'])
            rows, count = self._evaluate(np.unique(refined, axis=0), contour, observed, templates, cid)
            evaluations += count
            all_rows.extend(rows)
            seeds = self._distinct(all_rows, 12)
            ds /= 2
            dl /= 2
        return all_rows, evaluations

    @staticmethod
    def _distinct(rows, count):
        kept = []
        for row in sorted(rows, key=lambda r: (not (r['supported'] and r['score'] <= 1),
                                               r['score'], not r['supported'], abs(r['articulation_angle_deg']))):
            duplicate = False
            for other in kept:
                if row['contour_id'] != other['contour_id']:
                    continue
                a, b = row['_start'], other['_start']
                overlap = max(max(0., min(a+row['fraction'], b+shift+other['fraction']) - max(a, b+shift)) for shift in (-1, 0, 1))
                if overlap/min(row['fraction'], other['fraction']) >= .6:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(row)
                if len(kept) >= count:
                    break
        return kept

    def _evaluate(self, grid, contour, observed, templates, cid):
        p = self.parameters
        s, perimeter = _geometry(contour, True)
        n = p['sample_count']
        lengths = grid[:, 1]*perimeter
        positions = (grid[:, :1] + grid[:, 1:] * np.linspace(0, 1, n))*perimeter
        samples = _sample(contour, positions, True)
        # Test every traversed raw edge, not merely the resampled points.
        boundaries = np.r_[s, s[1:]+perimeter]
        first = np.searchsorted(boundaries, positions[:, 0], side='right')-1
        last = np.searchsorted(boundaries, positions[:, -1], side='left')
        missing = np.r_[0, np.cumsum(np.tile(~observed, 2))]
        supported = (missing[last]-missing[first] == 0) & (lengths >= p['min_arc_length_px'])
        all_rows = []
        for reverse in (False, True):
            q = samples[:, ::-1] if reverse else samples
            z = q[..., 0] + 1j*q[..., 1]
            center = z.mean(axis=1)
            centered = z-center[:, None]
            fine, coarse = _turn(q, 2), _turn(q, 5)
            resolved_arc = np.linalg.norm(np.diff(q, axis=1), axis=2).sum(axis=1)
            for angle, template, target_length in templates:
                tz = template[:, 0] + 1j*template[:, 1]
                tc = tz-tz.mean()
                factor = np.sum(centered*np.conj(tc), axis=1)/np.sum(np.abs(tc)**2)
                fitted = factor[:, None]*tc + center[:, None]
                distance = np.abs(z-fitted)/lengths[:, None]
                rms = np.sqrt(np.mean(distance**2, axis=1))
                maximum = np.max(distance, axis=1)
                tf, tb = _turn(template, 2), _turn(template, 5)
                fine_error = np.degrees(np.sqrt(np.mean(_angle_difference(fine, tf)**2, axis=1)))
                coarse_error = np.degrees(np.sqrt(np.mean(_angle_difference(coarse, tb)**2, axis=1)))
                excess = np.maximum(0., np.degrees((np.abs(fine).sum(axis=1)-np.abs(tf).sum())/2))
                template_resolved = np.linalg.norm(np.diff(template, axis=0), axis=1).sum()
                unresolved = np.maximum(0., 1-resolved_arc/lengths - (1-template_resolved/target_length))
                metrics = np.column_stack([rms, maximum, fine_error, coarse_error, excess, unresolved])
                names = ('rms', 'point_error', 'fine_turn_deg', 'coarse_turn_deg', 'excess_turn_deg', 'unresolved_arc_ratio')
                limits = np.array([p['max_'+name] for name in names])
                scores = np.max(metrics/limits, axis=1)
                # Preserve minima throughout the loop as well as the global best.
                # A dense cluster at one location must not hide another location
                # or exhaust memory for an uninformative template.
                order = np.argsort(scores, kind='stable')
                selected = set(order[:48].tolist())
                bins = np.floor(grid[:, 0]*32).astype(int)
                for group in np.unique(bins):
                    indices = np.flatnonzero(bins == group)
                    selected.add(int(indices[np.argmin(scores[indices])]))
                    visible = indices[supported[indices]]
                    if len(visible):
                        selected.add(int(visible[np.argmin(scores[visible])]))
                for i in sorted(selected):
                    start_position = positions[i, -1 if reverse else 0] % perimeter
                    end_position = positions[i, 0 if reverse else -1] % perimeter
                    a = q[i]
                    all_rows.append(dict(contour_id=int(cid), score=float(scores[i]),
                        supported=bool(supported[i]), passed=bool(scores[i] <= 1 and supported[i]),
                        reasons=([f'{name}_exceeds_limit' for name, value, limit in zip(names, metrics[i], limits) if value > limit] +
                                 ([] if supported[i] else ['unobserved_edges_or_insufficient_support'])),
                        metrics={name: float(value) for name, value in zip(names, metrics[i])},
                        fraction=float(grid[i, 1]), arc_length_px=float(lengths[i]),
                        start_arc_px=float(start_position), end_arc_px=float(end_position),
                        start_xy=a[0].tolist(), end_xy=a[-1].tolist(), reversed=reverse,
                        articulation_angle_deg=angle, scale=float(abs(factor[i])),
                        rotation_deg=float(np.degrees(np.angle(factor[i]))),
                        translation_xy=[float((center[i]-factor[i]*tz.mean()).real), float((center[i]-factor[i]*tz.mean()).imag)],
                        points_xy=a.tolist(), fitted_target_xy=np.column_stack([fitted[i].real, fitted[i].imag]).tolist(),
                        _start=float(grid[i, 0])))
        return all_rows, len(grid)*2*len(templates)
