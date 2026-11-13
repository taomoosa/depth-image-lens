"""ROI-free visible bridge evidence between two broad image-top entries.

A shape heuristic, not semantic gripper detection or proof of physical grasp.
"""
from dataclasses import dataclass, field
import numpy as np
from .depth import _numeric_image
from .geometry import bool_mask
from .mask_cleanup import _components8, _box_sum
from .oblique import _graph_components


@dataclass(frozen=True)
class TopEntryBridgeResult:
    bridge: str
    candidate_bridge: bool
    reasons: list
    entries: list
    metrics: dict
    thresholds: dict
    diagnostics: dict = field(default_factory=dict, repr=False)
    scope: str = 'visible_top_entry_connectivity_not_physical_grasp'
    validation_status: str = 'not_validated'

    def to_dict(self):
        return {k: v for k, v in vars(self).items() if k != 'diagnostics'}


def _erode(mask, radius):
    return mask.copy() if radius == 0 else _box_sum(mask, radius) == (2 * radius + 1) ** 2


def _close(mask, radius):
    if radius == 0:
        return mask.copy()
    return mask | _erode(_box_sum(mask, radius) > 0, radius)


def _spanning(parts, shape, seeds, min_area):
    result = np.zeros(shape, bool)
    for points in parts:
        if len(points) < min_area:
            continue
        ys, xs = points.T
        if all(seed[ys, xs].any() for seed in seeds):
            result[ys, xs] = True
    return result


class TopEntryBridgeDetector:
    """Find two broad top entries, then test a below-band depth-continuous path.

    The input mask may include both grippers and the object. The entry band must
    lie inside two separately visible grippers, above their joining material.
    Width support is a square erosion test, not a physical metric measurement.
    Gap closing produces only a candidate, never observed positive evidence.
    """

    def __init__(self, *, entry_band_height_px=8, min_entry_width_px=4,
                 min_entry_row_fraction=.6, min_entry_gap_px=2,
                 min_path_width_px=3, min_bridge_drop_px=3, min_area=20,
                 gap_radius_px=1, depth_jump_m=.05):
        positive = dict(entry_band_height_px=entry_band_height_px,
                        min_entry_width_px=min_entry_width_px,
                        min_entry_gap_px=min_entry_gap_px,
                        min_path_width_px=min_path_width_px, min_area=min_area)
        for name, value in positive.items():
            if type(value) is not int or value < 1:
                raise ValueError(name + ' must be a positive integer')
        for name, value in dict(min_bridge_drop_px=min_bridge_drop_px,
                                gap_radius_px=gap_radius_px).items():
            if type(value) is not int or value < 0:
                raise ValueError(name + ' must be a nonnegative integer')
        if min_path_width_px % 2 != 1:
            raise ValueError('min_path_width_px must be odd')
        if entry_band_height_px < min_path_width_px:
            raise ValueError('entry band must be at least min_path_width_px high')
        if gap_radius_px > 3:
            raise ValueError('gap_radius_px must be at most 3')
        if (not np.isfinite([min_entry_row_fraction, depth_jump_m]).all()
                or not 0 < min_entry_row_fraction <= 1 or depth_jump_m <= 0):
            raise ValueError('finite positive depth jump and row fraction in (0, 1] required')
        self.thresholds = {**positive, 'min_entry_row_fraction': float(min_entry_row_fraction),
                           'min_bridge_drop_px': min_bridge_drop_px,
                           'gap_radius_px': gap_radius_px, 'depth_jump_m': float(depth_jump_m)}

    def detect_mask(self, depth, foreground_mask, *, depth_unit,
                    observed_mask=None, exclude_mask=None, uncertain_mask=None,
                    mask_complete=False, diagnostics=False):
        """Use aligned HxW depth (explicit m/mm) and a boolean foreground mask.

        observed_mask marks actual depth observations, not mask completeness.
        uncertain_mask includes inferred repair pixels; they cannot prove a path.
        mask_complete=True explicitly attests that false mask pixels mean absence.
        Even then no is withheld for missing depth, weak paths and cropped paths.
        Input row zero is the top entry boundary; rotating/cropping is caller work.
        """
        if depth_unit not in ('m', 'mm'):
            raise ValueError('explicit m or mm depth_unit required')
        if type(mask_complete) is not bool or type(diagnostics) is not bool:
            raise ValueError('mask_complete and diagnostics must be bool')
        z = _numeric_image(depth, 'depth') * (.001 if depth_unit == 'mm' else 1.)
        supplied = bool_mask(foreground_mask, z.shape, 'foreground_mask')
        h, w = z.shape
        t = self.thresholds
        band = t['entry_band_height_px']
        if h <= band + t['min_bridge_drop_px'] + t['min_path_width_px'] or w < 2:
            raise ValueError('image must leave a below-entry bridge search')
        observed = np.ones(z.shape, bool) if observed_mask is None else bool_mask(observed_mask, z.shape, 'observed_mask')
        excluded = np.zeros(z.shape, bool) if exclude_mask is None else bool_mask(exclude_mask, z.shape, 'exclude_mask')
        uncertain = np.zeros(z.shape, bool) if uncertain_mask is None else bool_mask(uncertain_mask, z.shape, 'uncertain_mask')
        valid = np.isfinite(z) & (z > 0) & observed & ~uncertain
        silhouette = supplied & ~excluded
        upper = silhouette.copy(); upper[band:] = False
        labels, sizes = _components8(upper)
        entries = []; seeds = []
        for label in range(1, len(sizes)):
            part = labels == label
            ys, xs = np.nonzero(part)
            rows = np.count_nonzero(part[:band], axis=1)
            row_fraction = float(np.mean(rows >= t['min_entry_width_px']))
            if not (ys.min() == 0 and ys.max() == band - 1
                    and row_fraction >= t['min_entry_row_fraction']):
                continue
            entries.append(dict(x_min=int(xs.min()), x_max_exclusive=int(xs.max()) + 1,
                                center_x=float(np.median(xs)), pixels=int(sizes[label]),
                                supported_row_fraction=row_fraction))
            seed = part.copy(); seed[:max(0, band-t['min_path_width_px'])] = False
            seeds.append(seed)
        order = sorted(range(len(entries)), key=lambda i: entries[i]['center_x'])
        entries = [entries[i] for i in order]; seeds = [seeds[i] for i in order]
        metrics = dict(entry_count=len(entries), input_pixels=int(supplied.sum()),
                       inferred_pixels=0, observed_bridge_pixels=0)
        debug = {'entry_labels': labels, 'invalid_or_uncertain_mask': ~valid,
                 'excluded_mask': excluded.copy()} if diagnostics else {}

        def finish(value, candidate, reason):
            return TopEntryBridgeResult(value, bool(candidate), [reason], entries,
                                        metrics, dict(t), debug)

        if len(entries) != 2:
            return finish('unknown', False, 'requires_exactly_two_broad_separate_top_entries')
        left, right = entries
        gap = right['x_min'] - left['x_max_exclusive']
        metrics['entry_gap_px'] = gap
        if gap < t['min_entry_gap_px']:
            return finish('unknown', False, 'top_entries_overlap_or_too_close')
        yy, xx = np.indices(z.shape)
        # Allow the lower portions of entries, but forbid a shortcut across the
        # gap above the configured drop. Keep the search between the outer edges.
        margin = max(left['x_max_exclusive']-left['x_min'],
                     right['x_max_exclusive']-right['x_min'])
        search_left = max(0, left['x_min']-margin)
        search_right = min(w, right['x_max_exclusive']+margin)
        domain = ((yy >= band-t['min_path_width_px'])
                  & (xx >= search_left) & (xx < search_right))
        center_gap = (xx >= left['x_max_exclusive']) & (xx < right['x_min'])
        domain &= ~(center_gap & (yy < band+t['min_bridge_drop_px']))
        search_extent = domain.copy()
        domain &= ~excluded
        raw = silhouette & domain & valid
        parts, rejected, edge_count = _graph_components(raw, z, t['depth_jump_m'])
        path = _spanning(parts, z.shape, seeds, t['min_area'])
        radius = t['min_path_width_px'] // 2
        core = np.zeros(z.shape, bool)
        # Thickness must come from the same depth-connected surface, not from
        # adjacent but disconnected surfaces that merely fill the silhouette.
        for points in parts:
            ys, xs = points.T
            if len(points) < t['min_area'] or not all(seed[ys, xs].any() for seed in seeds):
                continue
            component = np.zeros(z.shape, bool); component[ys, xs] = True
            thick = _erode(component, radius)
            thick_parts, _, _ = _graph_components(thick, z, t['depth_jump_m'])
            core |= _spanning(thick_parts, z.shape, seeds, 1)
        supported = bool(core.any())
        metrics.update(observed_bridge_pixels=int(path.sum()),
                       robust_core_pixels=int(core.sum()), rejected_depth_edges=edge_count,
                       missing_search_pixels=int((search_extent & (~valid | excluded)).sum()))
        if diagnostics:
            debug.update(search_mask=domain, entry_seed_masks=np.stack(seeds),
                         observed_path_mask=path, robust_path_core_mask=core,
                         depth_rejected_edge_mask=rejected)
        if supported:
            return finish('yes', True, 'wide_observed_depth_continuous_path_between_top_entries')
        # Geometric near-connections may be noise, genuine gaps or missing data.
        # They are useful candidates but cannot turn into a positive observation.
        geometric = silhouette & domain
        repaired = _close(geometric, t['gap_radius_px']) & domain
        geo_labels, geo_sizes = _components8(repaired)
        candidate = np.zeros(z.shape, bool)
        entry_connected = np.zeros(z.shape, bool)
        for label in range(1, len(geo_sizes)):
            part = geo_labels == label
            if int((part & geometric).sum()) < t['min_area']:
                continue
            hits = [(part & seed).any() for seed in seeds]
            if any(hits):
                entry_connected |= part
            if all(hits):
                candidate |= part
        added = candidate & ~geometric
        metrics['inferred_pixels'] = int(added.sum())
        if diagnostics:
            debug.update(candidate_path_mask=candidate, inferred_gap_mask=added)
        if path.any():
            return finish('unknown', True, 'observed_connection_too_thin')
        if candidate.any():
            return finish('unknown', True, 'connection_requires_gap_or_unobserved_or_depth_discontinuous_pixels')
        # False mask pixels are not evidence of absence unless explicitly attested.
        boundary = np.zeros(z.shape, bool)
        boundary[-1] = True
        boundary[:, search_left] = True
        boundary[:, search_right-1] = True
        clipped = bool((silhouette & entry_connected & boundary & (yy >= band)).any())
        metrics['search_boundary_touched'] = clipped
        complete = (mask_complete and bool(valid[search_extent].all())
                    and not bool((excluded & search_extent).any()) and not clipped)
        if not complete:
            return finish('unknown', False, 'no_path_but_search_or_mask_incomplete')
        return finish('no', False, 'no_connection_in_complete_observed_search')
