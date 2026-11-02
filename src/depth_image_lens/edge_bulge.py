"""Local raised outer-edge evidence relative to the observed inward object surface."""
from dataclasses import dataclass, field
import numpy as np
from .boundary import _components, _inverse_plane
from .debug import DebugResultMixin, _opencv
from .depth import _numeric_image, _validate_depth_scale
from .geometry import CameraIntrinsics, bool_mask
from .pick import _clearance


def _distance_from(mask):
    cv = _opencv()
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return cv.distanceTransform((~mask).astype(np.uint8), cv.DIST_L2, cv.DIST_MASK_PRECISE)


def _reachable(seed, domain, radius):
    cv = _opencv()
    reached = seed & domain
    for _ in range(radius):
        grown = (cv.dilate(reached.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & domain
        if np.array_equal(grown, reached):
            break
        reached = grown
    return reached


@dataclass(frozen=True)
class EdgeBulgeResult(DebugResultMixin):
    state: str
    candidates: list
    regions: list
    rejections: list
    metrics: dict
    parameters: dict
    object_mask: np.ndarray = field(repr=False)
    candidate_mask: np.ndarray = field(repr=False)
    unknown_mask: np.ndarray = field(repr=False)
    evaluation_mask: np.ndarray = field(repr=False)
    reference_mask: np.ndarray = field(repr=False)
    excluded_mask: np.ndarray = field(repr=False)
    relative_height_m: np.ndarray = field(repr=False)
    diagnostics: dict = field(default_factory=dict, repr=False)
    scope: str = 'local_outer_edge_relief_not_absolute_height_or_physical_grasp_proof'

    def to_dict(self):
        return {k: v for k, v in vars(self).items() if not isinstance(v, np.ndarray) and k != 'diagnostics'}

    def reference_for_region(self, region_id):
        """Detached mask of valid inward fit inputs; detect(diagnostics=True) required."""
        if type(region_id) is not int or not any(r['id'] == region_id for r in self.regions):
            raise ValueError('region_id must identify a returned region')
        if 'context_pixels_by_region' not in self.diagnostics:
            raise ValueError('detect(diagnostics=True) required')
        mask = np.zeros(self.object_mask.shape, bool)
        mask.flat[self.diagnostics['context_pixels_by_region'][str(region_id)]] = True
        return mask

    def render_debug(self, color=None, *, color_order='RGB', **options):
        canvas = super().render_debug(color, color_order=color_order, **options)
        cv = _opencv()
        height = self.relative_height_m
        good = np.isfinite(height)
        limit = 4 * self.parameters['min_height_m']
        u = np.zeros(height.shape, np.uint8)
        u[good] = np.clip((height[good] / limit + 1) * 127.5, 0, 255).astype(np.uint8)
        rgb = cv.applyColorMap(u, cv.COLORMAP_TURBO)[..., ::-1].copy()
        rgb[~good] = (255, 0, 255)
        if color_order == 'BGR':
            rgb = rgb[..., ::-1]
        scale = min(1., 480 / max(height.shape))
        h, w = max(2, round(height.shape[0] * scale)), max(2, round(height.shape[1] * scale))
        tile = np.full((h + 48, canvas.shape[1], 3), 24, np.uint8)
        tile[48:48+h, :w] = cv.resize(rgb, (w, h), interpolation=cv.INTER_NEAREST)
        cv.putText(tile, f'Local inward reference residual +/-{limit:g} m (positive: toward camera)',
                   (5, 18), cv.FONT_HERSHEY_SIMPLEX, .4, (235, 235, 235), 1)
        cv.putText(tile, 'Magenta: no accepted reference fit; color saturation is not a height measurement.',
                   (5, 37), cv.FONT_HERSHEY_SIMPLEX, .4, (235, 235, 235), 1)
        return np.concatenate((canvas, tile), axis=0)


class OuterEdgeBulgeDetector:
    """Compare an outer object band to nearby inward samples, without outside depth.

    Uses registered optical-axis Z from a fixed overhead view. Each grid cell
    fits its own inward inverse-depth plane. A curved or poorly observed
    reference is unknown. The mask's outer silhouette is assumed supplied;
    holes and frame cuts are not interpreted as raised physical outer edges.
    """
    def __init__(self, *, band_px=16, edge_guard_px=2, attachment_px=6,
                 cell_px=24, context_px=24, context_gap_px=3,
                 min_context_points=30, min_context_fraction=.7,
                 max_reference_error_m=.003, min_height_m=.005,
                 depth_noise_m=.002, local_radius_px=1,
                 min_support_fraction=.6, local_depth_tolerance_m=.004,
                 min_region_area=25):
        integers = dict(band_px=(band_px, 2, 128), edge_guard_px=(edge_guard_px, 0, 32),
                        attachment_px=(attachment_px, 1, 128), cell_px=(cell_px, 8, 128),
                        context_px=(context_px, 2, 64), context_gap_px=(context_gap_px, 0, 32),
                        min_context_points=(min_context_points, 12, 100000),
                        local_radius_px=(local_radius_px, 1, 5), min_region_area=(min_region_area, 1, 1000000))
        for name, (value, low, high) in integers.items():
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{name} must be an integer in {low}..{high}')
        if not edge_guard_px < attachment_px <= band_px:
            raise ValueError('edge_guard_px < attachment_px <= band_px required')
        floats = dict(min_context_fraction=min_context_fraction, max_reference_error_m=max_reference_error_m,
                      min_height_m=min_height_m, depth_noise_m=depth_noise_m,
                      min_support_fraction=min_support_fraction, local_depth_tolerance_m=local_depth_tolerance_m)
        for name, value in floats.items():
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise ValueError(f'{name} must be a finite number')
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'{name} must be a finite number') from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
            floats[name] = value
        if not 0 < floats['min_context_fraction'] <= 1 or not .5 < floats['min_support_fraction'] <= 1:
            raise ValueError('context fraction in (0,1] and support fraction in (.5,1] required')
        if min(floats['max_reference_error_m'], floats['min_height_m'], floats['local_depth_tolerance_m']) <= 0:
            raise ValueError('positive height, reference error and depth tolerance required')
        self.parameters = dict((name, item[0]) for name, item in integers.items()) | floats

    def detect(self, depth, object_mask, *, depth_scale_m, registered_to_mask=False,
               intrinsics=None, observed_mask=None, depth_valid_mask=None, diagnostics=False):
        if registered_to_mask is not True:
            raise ValueError('registered_to_mask=True required')
        if type(diagnostics) is not bool:
            raise ValueError('diagnostics must be bool')
        if intrinsics is not None and not isinstance(intrinsics, CameraIntrinsics):
            raise ValueError('CameraIntrinsics required')
        _validate_depth_scale(depth_scale_m)
        with np.errstate(over='ignore', invalid='ignore'):
            z = _numeric_image(depth, 'depth') * depth_scale_m
        shape = z.shape
        obj = bool_mask(object_mask, shape, 'object_mask').copy()
        obs = np.ones(shape, bool) if observed_mask is None else bool_mask(observed_mask, shape, 'observed_mask')
        valid = np.isfinite(z) & (z > 0) & obs
        if depth_valid_mask is not None:
            valid &= bool_mask(depth_valid_mask, shape, 'depth_valid_mask')
        cv = _opencv(); p = self.parameters
        candidate = np.zeros(shape, bool); unknown = np.zeros(shape, bool)
        evaluation = np.zeros(shape, bool); reference = np.zeros(shape, bool)
        excluded = np.zeros(shape, bool); supported_all = np.zeros(shape, bool)
        outer_all = np.zeros(shape, bool); unsafe_all = np.zeros(shape, bool)
        residual = np.full(shape, np.nan); outer_distance = np.full(shape, np.nan)
        regions = []; candidates = []; rejections = []
        context_pixels = {}; source_regions = np.zeros(shape, np.int32)
        labels, parts = _components(obj)
        yy, xx = np.indices(shape)
        frame = np.zeros(shape, bool); frame[[0, -1], :] = True; frame[:, [0, -1]] = True
        obs_near = cv.erode(obs.astype(np.uint8), np.ones((3, 3), np.uint8),
                            borderType=cv.BORDER_CONSTANT, borderValue=0) > 0
        threshold = max(p['min_height_m'], 2 * p['depth_noise_m'])
        for component, _ in parts:
            comp = labels == component
            contours, _ = cv.findContours(comp.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            outer_u8 = np.zeros(shape, np.uint8); cv.drawContours(outer_u8, contours, -1, 1, 1)
            outer = outer_u8.astype(bool)
            dist = _distance_from(outer); outer_distance[comp] = dist[comp]
            unsafe = outer & (frame | ~obs_near)
            outer_all |= outer; unsafe_all |= unsafe
            band = comp & (dist <= p['band_px'])
            usable = band & (_clearance(comp) > p['edge_guard_px'] + 1) & (dist > p['edge_guard_px'])
            usable &= _distance_from(unsafe) > p['band_px']
            excluded |= band & ~usable
            evaluation |= usable
            unknown |= usable & ~valid
            # Spatial agreement is measured only within this original component.
            domain = comp & valid
            support_count = np.zeros(shape, np.int32); possible = np.zeros(shape, np.int32)
            r = p['local_radius_px']
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    a = (slice(max(0, -dy), min(shape[0], shape[0]-dy)), slice(max(0, -dx), min(shape[1], shape[1]-dx)))
                    b = (slice(max(0, dy), min(shape[0], shape[0]+dy)), slice(max(0, dx), min(shape[1], shape[1]+dx)))
                    possible[a] += comp[b]
                    with np.errstate(invalid='ignore'):
                        support_count[a] += comp[b] & valid[b] & (np.abs(z[a]-z[b]) <= p['local_depth_tolerance_m'])
            supported = comp & valid & (support_count >= np.maximum(5, p['min_support_fraction'] * possible))
            supported_all |= supported
            unknown |= usable & ~supported
            raised = np.zeros(shape, bool)
            ycells, xcells = np.nonzero(usable)
            cells = sorted(set(zip(ycells // p['cell_px'], xcells // p['cell_px'])))
            def tile_regions():
                for cy, cx in cells:
                    y0, x0 = int(cy*p['cell_px']), int(cx*p['cell_px'])
                    y1, x1 = min(shape[0], y0+p['cell_px']), min(shape[1], x0+p['cell_px'])
                    tile_labels, tile_parts = _components((usable & domain)[y0:y1,x0:x1])
                    for label, _ in tile_parts:
                        tile = np.zeros(shape, bool)
                        tile[y0:y1,x0:x1] = tile_labels == label
                        yield tile, y0, x0, y1, x1
            for tile, y0, x0, y1, x1 in tile_regions():
                item = dict(id=len(regions)+1, component_id=component, bbox_xyxy=[x0, y0, x1-1, y1-1],
                            status='unknown', reason=None, context_pixels=0, context_valid_fraction=0.,
                            context_fit_support=0, reference_error_p90_m=None)
                # Restrict growth to connected, observed object pixels. No bridge across holes.
                pad = p['context_px']; sy = slice(max(0,y0-pad), min(shape[0],y1+pad)); sx = slice(max(0,x0-pad), min(shape[1],x1+pad))
                reached = _reachable(tile[sy,sx], domain[sy,sx], pad)
                potential = _reachable(tile[sy,sx], (comp & obs)[sy,sx], pad)
                context = np.zeros(shape, bool)
                context[sy,sx] = potential & (dist[sy,sx] > p['band_px'] + p['context_gap_px'])
                context &= dist <= p['band_px'] + p['context_gap_px'] + pad
                context_valid = context & valid & supported
                context_valid[sy,sx] &= reached
                count = int(context_valid.sum()); fraction = count/max(1,int(context.sum()))
                item.update(context_pixels=count, context_valid_fraction=fraction)
                if diagnostics:
                    context_pixels[str(item['id'])] = np.flatnonzero(context_valid)
                try:
                    if count < p['min_context_points'] or fraction < p['min_context_fraction']:
                        raise ValueError('insufficient_inward_support')
                    coeff, fit_support = _inverse_plane(z, context_valid)
                    if fit_support < p['min_context_points']:
                        raise ValueError('insufficient_robust_inward_support')
                    inv = coeff[0]*xx + coeff[1]*yy + coeff[2]
                    measure = tile & valid
                    if np.any(inv[measure | context_valid] <= 1e-9):
                        raise ValueError('invalid_reference_prediction')
                    norm = None
                    if intrinsics is not None:
                        vector = np.array([coeff[0]*intrinsics.fx, coeff[1]*intrinsics.fy,
                                           coeff[0]*intrinsics.cx+coeff[1]*intrinsics.cy+coeff[2]])
                        norm = float(np.linalg.norm(vector))
                        if not np.isfinite(norm) or norm < 1e-12:
                            raise ValueError('invalid_reference_normal')
                    def relief(mask):
                        return 1./inv[mask]-z[mask] if norm is None else (1-z[mask]*inv[mask])/norm
                    error = float(np.quantile(np.abs(relief(context_valid)), .9))
                    item.update(context_fit_support=fit_support, reference_error_p90_m=error,
                                plane_inverse_z_coefficients=coeff.tolist())
                    if error > p['max_reference_error_m']:
                        raise ValueError('inward_reference_not_planar')
                    reference |= context_valid
                    residual[measure] = relief(measure)
                    source_regions[tile] = item['id']
                    local = tile & supported & (residual >= threshold + error)
                    raised |= local
                    item.update(status='computed', reason='local_inward_reference', raised_pixels=int(local.sum()))
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    unknown |= tile
                    item['reason'] = str(exc)
                regions.append(item)
            rl, rp = _components(raised)
            for key, area in rp:
                m = rl == key
                attached = bool(np.any(dist[m] <= p['attachment_px']))
                if area < p['min_region_area'] or not attached:
                    unknown |= m
                    ry, rx = np.nonzero(m)
                    rejections.append(dict(component_id=component, status='unknown', area_px=area,
                        reason='region_too_small' if area < p['min_region_area'] else 'not_attached_to_outer_edge',
                        bbox_xyxy=[int(rx.min()),int(ry.min()),int(rx.max()),int(ry.max())]))
                    continue
                y, x = np.nonzero(m)
                center = np.array([x.mean(), y.mean()]); xy = np.c_[x,y]
                point = xy[np.argmin(np.sum((xy-center)**2, axis=1))].tolist()
                candidates.append(dict(id=len(candidates)+1, component_id=component, status='edge_bulge_candidate',
                    reason='supported_camera_near_relief_against_inward_surface', area_px=area,
                    point_xy=point, bbox_xyxy=[int(x.min()),int(y.min()),int(x.max()),int(y.max())],
                    median_height_m=float(np.median(residual[m])), p95_height_m=float(np.quantile(residual[m],.95)),
                    inward_extent_px=float(dist[m].max()), touches_band_limit=bool(np.any(dist[m]>=p['band_px']-1)),
                    region_ids=[int(v) for v in np.unique(source_regions[m]) if v > 0]))
                candidate |= m
        unknown &= ~candidate
        metrics = dict(candidate_count=len(candidates), evaluated_pixels=int(evaluation.sum()),
                       unknown_pixels=int(unknown.sum()), excluded_pixels=int(excluded.sum()),
                       threshold_m=threshold, height_units='normal_m_toward_camera' if intrinsics is not None else 'camera_z_m_toward_camera',
                       warnings=['same_frame_inward_reference_not_table_height', 'coherent_depth_bias_can_mimic_relief',
                                 'broad_bulge_can_contaminate_reference', 'grid_and_pixel_scale_affect_results',
                                 'outer_silhouette_assumed_not_verified', 'excluded_edges_are_not_negative_detections'])
        state = 'candidate' if candidate.any() else 'unknown' if unknown.any() or not evaluation.any() else 'no_candidate'
        debug = dict(outer_boundary_mask=outer_all, unsafe_boundary_mask=unsafe_all,
                     supported_mask=supported_all, valid_depth_mask=valid & obj,
                     outer_distance_px=outer_distance, source_region_ids=source_regions,
                     context_pixels_by_region=context_pixels,
                     representative_points_xy=np.array([c['point_xy'] for c in candidates], dtype=int).reshape(-1,2)) if diagnostics else {}
        return EdgeBulgeResult(state,candidates,regions,rejections,metrics,dict(p),obj,candidate,unknown,evaluation,
                               reference,excluded,residual,debug)
