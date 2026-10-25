"""NumPy-only background-difference or supplied-mask shape heuristics. Not grasp proof."""
from collections import deque
from dataclasses import dataclass, field, replace
import numpy as np

__version__ = "0.2.0"


def _mask(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype != bool:
        raise ValueError(f"{name} must be a bool array with shape {shape}")
    return a



def _depth_arrays(depth, difference, depth_unit, difference_unit, difference_sign, background_valid, observed_mask, difference_mode="signed"):
    z = np.asarray(depth, dtype=float); diff = np.asarray(difference, dtype=float)
    if z.ndim != 2 or z.shape != diff.shape or min(z.shape) < 2:
        raise ValueError("depth/difference must be same HxW shape")
    if depth_unit not in ("m", "mm") or difference_unit != depth_unit:
        raise ValueError("declare the same m/mm unit for depth and difference")
    if difference_mode not in ("signed", "positive_foreground"):
        raise ValueError("difference_mode must be signed or positive_foreground")
    scale = .001 if depth_unit == "mm" else 1.
    if scale != 1.:
        z = z*scale; diff = diff*scale
    valid = np.isfinite(z) & (z > 0) & np.isfinite(diff)
    if difference_mode == "signed":
        if difference_sign not in ("background-current", "current-background"):
            raise ValueError("unknown difference_sign")
        diff = diff * (1 if difference_sign == "background-current" else -1)
        with np.errstate(invalid="ignore", over="ignore"):
            background = z + diff
        valid &= np.isfinite(background) & (background > 0)
    elif difference_sign not in (None, "background-current"):
        raise ValueError("positive_foreground uses supplied positives as-is; sign inversion is not allowed")
    # Masked positives are an input convention, not a physical background depth.
    # Finite zero/negative differences remain observed non-candidates here.
    if background_valid is not None: valid &= _mask(background_valid, z.shape, "background_valid")
    if observed_mask is not None: valid &= _mask(observed_mask, z.shape, "observed_mask")
    return z, diff, valid

@dataclass(frozen=True)
class BridgeDrapeResult:
    bridge: str
    large_sag: str
    drape: str
    reasons: dict
    features: dict
    foreground_mask: np.ndarray = field(repr=False)
    work_mask: np.ndarray = field(repr=False)
    invalid_mask: np.ndarray = field(repr=False)
    validation_status: str = "not_validated"
    scope: str = "visible_shape_heuristic_not_holding_or_tension"

    auto: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict, repr=False)
    stage_results: dict = field(default_factory=dict)

    def to_dict(self, *, include_masks=False):
        """Finite JSON scalars by default; optional bool-list masks can be large."""
        result = {k: getattr(self, k) for k in
                  ("bridge", "large_sag", "drape", "reasons", "features", "validation_status", "scope")}
        if self.stage_results: result["stage_results"] = self.stage_results
        if self.auto: result["auto"] = self.auto
        if self.diagnostics:
            result["diagnostics"] = {}
            for name, value in self.diagnostics.items():
                if isinstance(value, np.ndarray):
                    value = ({"shape": list(value.shape), "pixels": int(value.sum())}
                             if value.dtype == bool and not include_masks else value.tolist())
                result["diagnostics"][name] = value
        if include_masks:
            result.update({k: getattr(self, k).tolist() for k in
                           ("foreground_mask", "work_mask", "invalid_mask")})
        return result


class BridgeDrapeDetector:
    """Manual ROIs or bridge-first shape candidates, not semantic hand detection.

    Depth and difference must share pixels, units, depth meaning and valid-data
    conventions. Signed mode retains background-current geometry and sign options.
    Positive-foreground mode accepts supplied positive candidates as-is: zero is
    observed background/exclusion, not missing depth, and no background is rebuilt.
    Missing provenance must still be NaN or false in an optional validity mask.
    """
    def __init__(self, *, threshold_m=.02, min_area=20, anchor_band_px=3,
                 max_gap_px=0, sag_ratio=.15, drape_ratio=.25, static_hanging=True):
        if not np.isfinite([threshold_m, sag_ratio, drape_ratio]).all() or threshold_m < 0 or min(sag_ratio, drape_ratio) <= 0:
            raise ValueError("threshold must be nonnegative and ratios positive, all finite")
        if any(type(x) is not int for x in (min_area, anchor_band_px, max_gap_px)) or min_area < 1 or anchor_band_px < 1 or not 0 <= max_gap_px <= 2:
            raise ValueError("positive integer area/band and gap0..2 pixels required")
        if type(static_hanging) is not bool: raise ValueError("static_hanging must be bool")
        self.static_hanging = static_hanging
        self.threshold_m = float(threshold_m)
        self.min_area = min_area
        self.anchor_band_px = anchor_band_px
        self.max_gap_px = max_gap_px
        self.sag_ratio = float(sag_ratio)
        self.drape_ratio = float(drape_ratio)

    def _components(self, foreground, traversable, depth=None, depth_jump_m=.10):
        """Eight-neighbour paths; optional tiny gaps only through observed pixels.

        No new foreground is created. Invalid/excluded pixels are never jumped.
        """
        # Floating arithmetic tolerance only, not a sensor-noise margin.
        roundoff_factor = 4*np.finfo(float).eps
        remaining = foreground.copy(); h, w = remaining.shape
        for sy, sx in zip(*np.nonzero(foreground)):
            if not remaining[sy, sx]:
                continue
            queue = deque([(int(sy), int(sx))]); remaining[sy, sx] = False
            pixels = []; gaps = []; discontinuous = False
            while queue:
                y, x = queue.popleft(); pixels.append((y, x))
                for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                    previous = (y,x); path_gap = []; exceeds_depth_limit = False
                    for step in range(1, self.max_gap_px + 2):
                        yy, xx = y + dy*step, x + dx*step
                        if not (0 <= yy < h and 0 <= xx < w) or not traversable[yy, xx]:
                            break
                        if depth is not None:
                            current_z,previous_z = float(depth[yy,xx]),float(depth[previous])
                            tolerance = roundoff_factor*max(abs(current_z),abs(previous_z),depth_jump_m)
                            exceeds_depth_limit |= abs(current_z-previous_z) > depth_jump_m+tolerance
                        previous = (yy,xx)
                        if foreground[yy, xx]:
                            # An unsafe optional shortcut is ignored; a valid route
                            # around the hole may still connect the observed pixels.
                            if step > 1 and exceeds_depth_limit:
                                break
                            discontinuous |= exceeds_depth_limit
                            if remaining[yy, xx]:
                                remaining[yy, xx] = False; queue.append((yy, xx))
                                gaps.extend(path_gap)
                            break
                        path_gap.append((yy,xx))
            if len(pixels) >= self.min_area:
                yield np.asarray(pixels, dtype=int), np.asarray(gaps,dtype=int).reshape(-1,2), discontinuous

    def detect_16uc1(self, depth, difference, hand_rois=None, *, depth_scale_m=0.001, **options):
        """Convenience for clipped positive uint16 counts; no ROS dependency.

        One scale applies to both arrays. Omit hand_rois for automatic candidates.
        Pass the preparation helper's validity mask when background had holes.
        """
        from .depth import depth_pair_16uc1_to_meters
        if set(options) & {"depth_unit", "difference_unit", "difference_sign", "difference_mode"}:
            raise ValueError("16UC1 convenience fixes metric units and positive_foreground mode")
        current, delta = depth_pair_16uc1_to_meters(depth, difference, depth_scale_m=depth_scale_m)
        if hand_rois is None:
            return self.detect_auto(current, delta, difference_mode="positive_foreground", **options)
        return self.detect(current, delta, hand_rois, difference_mode="positive_foreground", **options)

    def detect_auto(self, depth, difference, *, upper_band=None, min_span_px=30,
                    max_height_delta_px=12, depth_jump_m=.10, down=(0,1), diagnostics=False, **options):
        """Bridge first, then drape necks. All ROIs are unverified image candidates.

        upper_band=(v_min,v_max) is along normalized image down. Its lower edge
        separates upper arm clutter from the deformable-object region. Default: top quarter
        of the image. A poor boundary can miss the deformable object; this is not semantic detection.
        options accepts detect() units, validity masks, arm exclusion and work_roi, but not hand ROIs
        or visibility declarations. Missing opposite evidence stays unknown.
        """
        if type(diagnostics) is not bool: raise ValueError("diagnostics must be bool")
        allowed={"depth_unit","difference_unit","difference_sign","difference_mode","background_valid","observed_mask","arm_mask","work_roi"}
        if set(options)-allowed:raise ValueError("unsupported auto option; manual ROIs use detect()")
        if not np.isfinite([min_span_px,max_height_delta_px,depth_jump_m]).all() or min(min_span_px,max_height_delta_px,depth_jump_m)<=0:
            raise ValueError("positive finite auto thresholds required")
        z,diff,valid=_depth_arrays(depth,difference,options.get("depth_unit","m"),options.get("difference_unit","m"),
            options.get("difference_sign","background-current"),options.get("background_valid"),options.get("observed_mask"),options.get("difference_mode","signed"))
        d=np.array(down,float,copy=True)
        if d.shape!=(2,) or not np.isfinite(d).all() or not np.isfinite(np.linalg.norm(d)) or np.linalg.norm(d)<1e-9:
            raise ValueError("finite nonzero down required")
        d/=np.linalg.norm(d);axis=np.array([d[1],-d[0]])
        yy,xx=np.indices(z.shape);u=xx*axis[0]+yy*axis[1];v=xx*d[0]+yy*d[1]
        band=np.array((v.min(),v.min()+.25*np.ptp(v)) if upper_band is None else upper_band,float)
        if band.shape!=(2,) or not np.isfinite(band).all() or band[0]>=band[1]:raise ValueError("ordered finite upper_band required")
        if not v.min()<band[1]<v.max():raise ValueError("upper_band lower edge must leave a visible below-band search")
        arm=np.zeros(z.shape,bool) if options.get('arm_mask') is None else _mask(options['arm_mask'],z.shape,'arm_mask')
        roi=np.ones(z.shape,bool) if options.get('work_roi') is None else _mask(options['work_roi'],z.shape,'work_roi')
        fg=valid&(diff>0)&(diff>self.threshold_m);usable=valid&~arm&roi
        work=fg&usable&(v>band[1])
        meta={"stage":"search","identity":"image transverse left/right, not robot IDs","upper_band":band.tolist(),
              "hand_roi_candidates":{},"opposite_search_region":None,"physical_contact_verified":False,
              "candidate_knowledge":"shape and same-height foreground only; no semantic hand model"}
        def uncertain(reason, bridge="unknown", drape="unknown"):
            diag = {}
            if diagnostics:
                diag = {"down": d.tolist(), "candidate_search_mask": usable & (v > band[1])}
                opposite = meta.get("opposite_search_region")
                if opposite:
                    lo, hi = opposite["u_outside"]; top, bottom = opposite["v_range"]
                    diag["opposite_search_mask"] = ((u <= lo) | (u >= hi)) & (v >= top) & (v <= bottom)
            result = BridgeDrapeResult(bridge,"unknown",drape,dict.fromkeys(("bridge","large_sag","drape"),reason),
                {"difference_mode":options.get("difference_mode","signed"),"sag_ratio":None,"span_coverage":None,"valid_fraction":float(valid.mean()),**meta.get('observed_features',{})},fg,work,~valid,auto=meta,diagnostics=diag)
            return self._with_stage_results(result,z,np.isfinite(diff)&(diff>self.threshold_m)&(diff>0),valid,None,
                arm_mask=arm,work_roi=roi,hand_visibility=('hidden','hidden'),down=down,
                depth_jump_m=depth_jump_m,mask_complete=True,diagnostics=diagnostics)
        # An upper cut cannot be repaired by assuming invisible hand positions.
        # With tilted image down, a projected minimum covers only a corner.
        # Any upstream image edge can cut off supports, including away from it.
        upper_edge = np.zeros(z.shape, bool)
        if d[1] > 0: upper_edge[:2, :] = True
        if d[1] < 0: upper_edge[-2:, :] = True
        if d[0] > 0: upper_edge[:, :2] = True
        if d[0] < 0: upper_edge[:, -2:] = True
        if np.any(fg & usable & upper_edge):
            # Apply the same minimum-area rule before treating clipping as evidence.
            # Isolated threshold specks must not veto an otherwise visible candidate.
            for pixels, _, _ in self._components(fg & usable, usable):
                if np.any(upper_edge[pixels[:, 0], pixels[:, 1]]):
                    meta['stage']='upper_clipped';return uncertain('upper_foreground_clipped')
        part_data=list(self._components(work,usable&(v>band[1]),z,depth_jump_m))
        part_data.sort(key=lambda item:len(item[0]),reverse=True);part_data=part_data[:4]
        parts=[item[0] for item in part_data]
        quality={id(p):(gaps,bad) for p,gaps,bad in part_data}
        meta.update(component_count_considered=len(parts),candidate_evaluations=[],
                    selection_rule='up to four largest components, then up to two widest eligible spans')
        def point(p, edge=None):
            ys,xs=p.T;cu,cv=u[ys,xs],v[ys,xs]
            pick=np.ones(len(p),bool)
            if edge=='left':pick=cu<=cu.min()+max(2,.1*np.ptp(cu))
            if edge=='right':pick=cu>=cu.max()-max(2,.1*np.ptp(cu))
            top=float(cv[pick].min());near=pick&(cv<=top+2)
            return float(np.median(cu[near])),top
        def box(pt):
            pu,pv=pt;r=max(3,self.anchor_band_px+2)
            corners=np.array([pu*axis+pv*d-r*axis,pu*axis+pv*d+r*axis,
                              pu*axis+(pv-2*r)*d-r*axis,pu*axis+(pv-2*r)*d+r*axis])
            return [int(np.floor(corners[:,0].min())),int(np.floor(corners[:,1].min())),
                    int(np.ceil(corners[:,0].max()))+1,int(np.ceil(corners[:,1].max()))+1]
        def run_pair(a,b,stage):
            a,b=sorted((a,b));boxes={'left':box(a),'right':box(b)}
            meta.update(stage=stage,hand_roi_candidates=boxes,anchor_points_uv=[list(a),list(b)])
            r=self.detect(depth,difference,boxes,down=down,depth_jump_m=depth_jump_m,diagnostics=diagnostics,**options)
            local={}
            for name,pt in [('left',a),('right',b)]:
                region=work&(np.abs(u-pt[0])<=.25*(b[0]-a[0]))
                local['image_'+name+'_local_hanging_extent_px']=float(max(0.,np.quantile(v[region]-pt[1],.95))) if region.any() else 0.
            diag = dict(r.diagnostics)
            if diagnostics: diag["candidate_search_mask"] = usable & (v > band[1])
            return replace(r,features={**r.features,**local},auto=meta.copy(),diagnostics=diag)
        bridge_candidates=[];height_rejected=False
        upper_u=u[work&(v<=band[1]+max_height_delta_px)]
        upper_extent=np.quantile(upper_u,[.02,.98]) if upper_u.size else None
        for p in parts:
            a,b=point(p,'left'),point(p,'right')
            if b[0]-a[0]<min_span_px:continue
            # A disconnected half of a broken bridge must not become a smaller
            # bridge by inventing an inner anchor. Other upper foreground vetoes it.
            if upper_extent is not None:
                tolerance=max(5.,.15*float(np.ptp(upper_extent)))
                if a[0]>upper_extent[0]+tolerance or b[0]<upper_extent[1]-tolerance:continue
            if abs(a[1]-b[1])>max_height_delta_px or max(a[1],b[1])>band[1]+max_height_delta_px:
                height_rejected=True;continue
            # This is a connected *below-band* path, never the upper arm component.
            bridge_candidates.append((b[0]-a[0],a,b,p))
        depth_rejected=False; rejected_pair=None
        for rank,(candidate_span,a,b,p) in enumerate(sorted(bridge_candidates,key=lambda x:-x[0])[:2],1):
            gaps,bad=quality[id(p)]
            evaluation={'rank':rank,'span_px':float(candidate_span),'pixels':len(p),
                        'gap_inference_used':bool(len(gaps)), 'depth_continuous':not bad}
            meta['candidate_evaluations'].append(evaluation)
            # A bad branch is evidence about local edges, not every route between
            # the proposed bands. The shared feature pass checks observed routes.
            r=run_pair(a,b,'bridge_first')
            evaluation.update(observed_route=r.stage_results['observed_bridge']['value'],
                rejected_local_edges=r.stage_results['depth_connectivity']['rejected_edge_count'])
            if r.bridge=='yes':
                evaluation['decision']='selected'
                return r
            if bad:
                evaluation['decision']='rejected_depth_discontinuity';depth_rejected=True
                if rejected_pair is None:rejected_pair=r
            else:evaluation['decision']='manual_check_'+r.bridge
        if depth_rejected:
            # Retain candidate bands and their independently computed measurements,
            # while allowing all ranked bridge alternatives above to be tested.
            return replace(rejected_pair,auto={**rejected_pair.auto,'stage':'depth_discontinuous',
                'candidate_evaluations':meta['candidate_evaluations']})
        if height_rejected and not bridge_candidates:
            meta['stage']='height_assumption_violation';return uncertain('upper_extrema_not_at_similar_height')
        necks=[]
        for p in parts:
            if quality[id(p)][1]:continue
            a=point(p);ys,xs=p.T;extent=float(np.quantile(v[ys,xs],.95)-a[1])
            if extent>=min_span_px*.5 and a[1]<=band[1]+max_height_delta_px:
                local=(np.abs(u[ys,xs]-a[0])<=5)&(v[ys,xs]<=a[1]+4)
                spread=float(np.ptp(z[ys[local],xs[local]])) if local.any() else None
                if spread is not None and spread<=depth_jump_m:necks.append((a,extent,spread))
        if len(necks)>=2:
            a,b=sorted(necks,key=lambda n:n[0][0])[::len(necks)-1]
            if b[0][0]-a[0][0]>=min_span_px and abs(a[0][1]-b[0][1])<=max_height_delta_px:
                return run_pair(a[0],b[0],'separate_drapes')
        if len(necks)==1:
            a,extent,spread=necks[0]
            meta['observed_features']={'observed_hanging_extent_px':extent,'neck_depth_spread_m':spread}
            opposite=fg&usable&(np.abs(v-a[1])<=max_height_delta_px)&(np.abs(u-a[0])>=min_span_px)
            others=[p for p,_,bad in self._components(opposite,usable,z,depth_jump_m) if not bad]
            if len(others)==1:
                ys,xs=others[0].T;b=(float(np.median(u[ys,xs])),float(v[ys,xs].max()))
                if abs(b[1]-a[1])<=max_height_delta_px:return run_pair(a,b,'drape_then_opposite_foreground')
            meta.update(stage='one_neck_opposite_unobserved',hand_roi_candidates={'observed':box(a)},
                opposite_search_region={'u_outside':[a[0]-min_span_px,a[0]+min_span_px],
                                        'v_range':[a[1]-max_height_delta_px,a[1]+max_height_delta_px]})
            return uncertain('one_neck_only_no_opposite_hand_invented')
        meta['stage']='no_supported_pair'
        if not fg.any() and valid.all():return uncertain('no_foreground_in_observed_image','no','neither')
        # No below-band deformable-object candidate; upper foreground may be arms only.
        if not parts and np.all((valid&~arm&roi)[v>band[1]]):return uncertain('no_below_band_work_path','no','neither')
        return uncertain('no_reliable_anchor_pair')

    def detect(self, depth, difference, hand_rois, *, depth_unit="m", difference_unit="m",
               difference_sign="background-current", difference_mode="signed", background_valid=None, observed_mask=None,
               arm_mask=None, work_roi=None, hand_visibility=("visible", "visible"), down=(0,1), diagnostics=False, depth_jump_m=.10, anchor_policy="complete"):
        """ROIs: {'left':(x0,y0,x1,y1), 'right':(...)} with exclusive x1/y1.

        Use coarse hand boxes whose down-side edges are near the material contact.
        A thin lower band is retained as anchor; hand interiors are excluded.
        Visibility is a caller declaration ('visible', 'partial', 'hidden'), not
        inferred from valid depth. Mark occlusion in observed_mask as well.
        Units m/mm must agree. Threshold always uses metres. down is image-space.
        Optional local_observed anchor policy admits only positive paths with a
        fully observed 3x3 patch in each anchor of the same component. It requires
        max_gap_px=0 and does not relax negative, drape or sag completeness.
        Candidate support does not establish physical contact.
        """
        if anchor_policy not in ("complete", "local_observed"):
            raise ValueError("anchor_policy must be complete or local_observed")
        if anchor_policy == "local_observed" and self.max_gap_px:
            raise ValueError("local_observed requires max_gap_px=0")
        if type(diagnostics) is not bool: raise ValueError("diagnostics must be bool")
        if not np.isfinite(depth_jump_m) or depth_jump_m <= 0:
            raise ValueError("depth_jump_m must be positive and finite")
        z, diff, valid = _depth_arrays(depth, difference, depth_unit, difference_unit, difference_sign, background_valid, observed_mask, difference_mode)
        foreground = valid & (diff > 0) & (diff > self.threshold_m)
        result = self._detect_mask_geometry(z, valid, foreground, hand_rois, difference_mode=difference_mode,
            arm_mask=arm_mask, work_roi=work_roi, hand_visibility=hand_visibility, down=down,
            diagnostics=diagnostics, depth_jump_m=depth_jump_m, anchor_policy=anchor_policy)
        return self._with_stage_results(result,z,np.isfinite(diff)&(diff>0)&(diff>self.threshold_m),valid,hand_rois,
            arm_mask=arm_mask,work_roi=work_roi,hand_visibility=hand_visibility,down=down,
            depth_jump_m=depth_jump_m,mask_complete=True,diagnostics=diagnostics)

    def detect_mask(self, depth, foreground_mask, hand_rois=None, *, depth_unit,
                    observed_mask=None, arm_mask=None, work_roi=None,
                    hand_visibility=("visible", "visible"), down=(0, 1), diagnostics=False,
                    depth_jump_m=.10, anchor_policy="complete", mask_complete=False):
        """Explicit supplied-mask geometry; no background or numeric difference.

        depth_unit must be m or mm. True foreground is caller-supplied object
        evidence; false pixels are excluded, not invented background measurements.
        Sensor invalidity is retained separately. Two candidate bands are needed for
        an observed bridge; missing anchors do not suppress silhouette measurements.
        mask_complete=False withholds absence claims, with warnings on measured sag.
        Masks/ROIs are assumptions, never independent ground truth or hand detection.
        """
        from .depth import _numeric_image
        if depth_unit not in ('m', 'mm'): raise ValueError('explicit m or mm depth_unit required')
        if type(mask_complete) is not bool or type(diagnostics) is not bool:
            raise ValueError('mask_complete and diagnostics must be bool')
        if anchor_policy not in ('complete', 'local_observed'):
            raise ValueError('anchor_policy must be complete or local_observed')
        if self.max_gap_px:
            raise ValueError('supplied-mask geometry requires max_gap_px=0')
        if not np.isfinite(depth_jump_m) or depth_jump_m <= 0:
            raise ValueError('depth_jump_m must be positive and finite')
        z = _numeric_image(depth, 'depth')
        if depth_unit == 'mm': z = z * .001
        supplied = _mask(foreground_mask, z.shape, 'foreground_mask')
        sensor_valid = np.isfinite(z) & (z > 0)
        observed = np.ones(z.shape, bool) if observed_mask is None else _mask(observed_mask, z.shape, 'observed_mask')
        valid = sensor_valid & observed
        foreground = supplied & valid
        if hand_rois is None:
            arms = np.zeros(z.shape, bool) if arm_mask is None else _mask(arm_mask, z.shape, 'arm_mask')
            roi = np.ones(z.shape, bool) if work_roi is None else _mask(work_roi, z.shape, 'work_roi')
            result = BridgeDrapeResult('unknown', 'unknown', 'unknown',
                dict.fromkeys(('bridge', 'large_sag', 'drape'), 'supplied_mask_requires_two_anchor_rois'),
                {'difference_mode': 'supplied_mask', 'valid_fraction': float(valid.mean())},
                foreground, foreground & ~arms & roi, ~valid)
        else:
            result = self._detect_mask_geometry(z, valid, foreground, hand_rois, difference_mode='supplied_mask',
                arm_mask=arm_mask, work_roi=work_roi, hand_visibility=hand_visibility, down=down,
                diagnostics=diagnostics, depth_jump_m=depth_jump_m, anchor_policy=anchor_policy)
        original = {'bridge': result.bridge, 'large_sag': result.large_sag, 'drape': result.drape}
        bridge, sag, drape = result.bridge, result.large_sag, result.drape
        reasons = dict(result.reasons)
        if not mask_complete:
            if bridge == 'no': bridge = 'unknown'; reasons['bridge'] = 'supplied_mask_completeness_unverified'
            if drape in ('left', 'right', 'neither'):
                drape = 'unknown'; reasons['drape'] = 'supplied_mask_opposite_absence_unverified'
            if sag != 'unknown': sag = 'unknown'; reasons['large_sag'] = 'supplied_mask_upper_envelope_unverified'
        features = {**result.features, 'input_mode': 'supplied_object_mask', 'measured_background_used': False,
                    'mask_completeness_caller_attested': mask_complete, 'physical_contact_verified': False,
                    'supplied_mask_pixels': int(supplied.sum()), 'supplied_mask_invalid_pixels': int((supplied & ~valid).sum()),
                    'supplied_mask_geometry_before_guard': original}
        diag = dict(result.diagnostics)
        if diagnostics:
            diag['sensor_invalid_mask'] = ~sensor_valid
            diag['caller_unobserved_mask'] = ~observed
            diag['supplied_mask_excluded_mask'] = valid & ~supplied
            diag['supplied_mask_invalid_mask'] = supplied & ~valid
            diag['sag_path_used'] = sag != 'unknown'
        result = replace(result, bridge=bridge, large_sag=sag, drape=drape, reasons=reasons, features=features, diagnostics=diag)
        return self._with_stage_results(result,z,supplied,valid,hand_rois,arm_mask=arm_mask,work_roi=work_roi,
            hand_visibility=hand_visibility,down=down,depth_jump_m=depth_jump_m,mask_complete=mask_complete,diagnostics=diagnostics)

    def _with_stage_results(self,result,z,candidates,valid,hand_rois,*,arm_mask,work_roi,
                            hand_visibility,down,depth_jump_m,mask_complete,diagnostics):
        from .oblique import ObliqueShapeAnalyzer
        arm=np.zeros(z.shape,bool) if arm_mask is None else _mask(arm_mask,z.shape,'arm_mask')
        roi=np.ones(z.shape,bool) if work_roi is None else _mask(work_roi,z.shape,'work_roi')
        stages,trace=ObliqueShapeAnalyzer().analyze(z,candidates,valid,arm_mask=arm,work_roi=roi,
            hand_rois=hand_rois,down=down,hand_visibility=hand_visibility,anchor_band_px=self.anchor_band_px,
            min_area=self.min_area,depth_jump_m=depth_jump_m,sag_ratio=self.sag_ratio,drape_ratio=self.drape_ratio,
            mask_complete=mask_complete,static_hanging=self.static_hanging,diagnostics=diagnostics)
        before={'bridge':result.bridge,'large_sag':result.large_sag,'drape':result.drape,'reasons':dict(result.reasons)}
        bridge,sag,drape=result.bridge,result.large_sag,result.drape;reasons=dict(result.reasons)
        features=dict(result.features);diag={**result.diagnostics,**trace}
        if stages['observed_bridge']['value']=='yes':
            bridge='yes';reasons['bridge']=stages['observed_bridge']['reason']
            if diagnostics:diag['bridge_corridor_mask']=trace['feature_observed_bridge_mask']
        elif bridge=='no' and mask_complete:
            stages['observed_bridge'].update(status='computed',value='no',reason='no_path_in_complete_legacy_search')
        if stages['sag']['status']=='computed' and stages['sag']['value']!='unknown':
            sag=stages['sag']['value'];reasons['large_sag']='measured_visible_envelope_vs_candidate_chord'
            features.update(sag_ratio=stages['sag']['ratio'],span_coverage=stages['sag']['observed_span_fraction'])
            if diagnostics:
                diag['sag_path_used']=True
                if 'feature_sag_path_xy' in trace:diag['sag_path_xy']=trace['feature_sag_path_xy']
        if diagnostics:
            for name in ('left','right'):
                key='feature_'+name+'_drape_component_mask'
                if key in trace:diag[name+'_drape_component_mask']=trace[key]
        attachments=stages['attachment_drape']['candidates']
        if len(attachments)==2 and all(a.get('hangs') is True for a in attachments):
            drape='both';reasons['drape']='measured_extent_from_both_candidate_bands'
        features.update(conservative_decisions=before,independent_feature_analysis=True,
                        result_scope='visible_geometry_not_grasp',static_hanging_assumption=self.static_hanging)
        return replace(result,bridge=bridge,large_sag=sag,drape=drape,reasons=reasons,
                       features=features,diagnostics=diag,stage_results=stages)

    def _detect_mask_geometry(self, z, valid, foreground, hand_rois, *, difference_mode,
                              arm_mask, work_roi, hand_visibility, down, diagnostics,
                              depth_jump_m, anchor_policy):
        if not isinstance(hand_rois, dict) or set(hand_rois) != {"left", "right"}:
            raise ValueError("hand_rois needs explicit left and right boxes")
        if len(hand_visibility) != 2 or any(x not in ("visible", "partial", "hidden") for x in hand_visibility):
            raise ValueError("two caller-declared hand visibility values required")
        d = np.array(down, dtype=float, copy=True)
        if d.shape != (2,) or not np.isfinite(d).all() or not np.isfinite(np.linalg.norm(d)) or np.linalg.norm(d) < 1e-9:
            raise ValueError("down must be a finite nonzero image vector")
        d /= np.linalg.norm(d); axis = np.array([d[1], -d[0]])
        arms = np.zeros(z.shape, bool) if arm_mask is None else _mask(arm_mask, z.shape, "arm_mask")
        roi = np.ones(z.shape, bool) if work_roi is None else _mask(work_roi, z.shape, "work_roi")
        h, w = z.shape; yy, xx = np.indices(z.shape)
        u = xx*axis[0] + yy*axis[1]; v = xx*d[0] + yy*d[1]
        anchors = []; points = []; hand_interiors = np.zeros(z.shape, bool); inside = []
        for name in ("left", "right"):
            box = np.asarray(hand_rois[name], dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError("finite nonempty xyxy hand boxes required")
            x0,y0,x1,y1 = box
            corners = np.array([[x0,y0],[x1-1,y0],[x0,y1-1],[x1-1,y1-1]])
            uc = corners@axis; vc = corners@d; pu = float(uc.mean()); pv = float(vc.max())
            points.append((pu,pv)); inside.append(x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h)
            anchor = (u >= uc.min()) & (u <= uc.max()) & (v >= pv-self.anchor_band_px) & (v <= pv+self.anchor_band_px)
            anchors.append(anchor)
            hand_interiors |= (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1) & ~anchor
        (lu,lv),(ru,rv) = points; span = abs(ru-lu)
        if not np.isfinite([lu,lv,ru,rv,span]).all():
            raise ValueError("hand ROI coordinates exceed finite geometry range")
        empty = np.zeros(z.shape, bool)
        diag = ({"down": d.tolist(), "left_anchor_mask": anchors[0], "right_anchor_mask": anchors[1],
                 "sag_chord_xy": np.asarray(points) @ np.stack((axis, d)), "sag_path_used": False}
                if diagnostics else {})
        features = {"difference_mode":difference_mode, "sag_ratio":None, "span_coverage":None, "span_px":float(span),
                    "left_hanging_extent_px":None, "right_hanging_extent_px":None,
                    "left_hanging_width_px":None, "right_hanging_width_px":None,
                    "valid_fraction":float(valid.mean()), "foreground_pixels":int(foreground.sum()),
                    "gap_tolerance_px":self.max_gap_px, "gap_used":False}
        if span < 8 or np.any(anchors[0] & anchors[1]):
            return BridgeDrapeResult("unknown","unknown","unknown",dict.fromkeys(("bridge","large_sag","drape"),"anchors_overlap_or_near_vertical"),features,foreground,empty,~valid,diagnostics=diag)
        baseline = lv + (u-lu)*(rv-lv)/(ru-lu)
        margin = span*.25
        search = roi & (u >= min(lu,ru)-margin) & (u <= max(lu,ru)+margin) & (v >= baseline-self.anchor_band_px)
        search &= ~hand_interiors
        usable = search & valid & ~arms
        work = foreground & usable
        if diagnostics: diag["search_mask"] = search
        observable = [bool(inside[i] and hand_visibility[i] == "visible" and anchors[i].any()
                      and np.all((valid & ~arms & roi)[anchors[i]])) for i in range(2)]
        complete = all(observable) and bool(np.all((valid & ~arms)[search]))
        components = []; kept = np.zeros(z.shape, bool); spans = []; clipped = False
        local_spans = []; local_support = np.zeros(z.shape, bool)
        inferred_gap = np.zeros(z.shape,bool); rejected_depth = np.zeros(z.shape,bool)
        for pixels,gaps,bad in self._components(work, usable,z,depth_jump_m):
            ys,xs = pixels.T; kept[ys,xs] = True
            hits = [int(anchors[i][ys,xs].sum()) >= 3 for i in range(2)]
            if not any(hits):continue
            if bad:
                rejected_depth[ys,xs]=True;complete=False
                continue
            cu,cv = u[ys,xs],v[ys,xs]
            # Below-anchor contact with search/image boundary can hide a path.
            below = cv > baseline[ys,xs]+self.anchor_band_px
            boundary = (ys==0)|(ys==h-1)|(xs==0)|(xs==w-1)|(cu <= min(lu,ru)-margin+1)|(cu >= max(lu,ru)+margin-1)
            if work_roi is not None:
                pad=np.pad(roi,1); edge=np.zeros(z.shape,bool)
                for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                    edge |= ~pad[1+dy:1+dy+h,1+dx:1+dx+w]
                boundary |= edge[ys,xs]
            clipped |= bool(np.any(boundary & below))
            components.append((pixels,hits))
            if all(hits):
                if anchor_policy == "local_observed":
                    # Positive-only alternative: an actually observed 3x3 patch at
                    # each anchor in this same component. Never restore invalid data,
                    # visibility, crop, exclusions, or infer a connection across holes.
                    component = np.zeros(z.shape, bool); component[ys, xs] = True
                    supports = []
                    for anchor in anchors:
                        patch = component & anchor & usable
                        padded = np.pad(patch, 1)
                        interior = patch.copy()
                        for dy in range(3):
                            for dx in range(3):
                                interior &= padded[dy:dy+h, dx:dx+w]
                        supports.append(interior)
                    if all(inside) and all(x == "visible" for x in hand_visibility) and all(x.any() for x in supports) and not len(gaps):
                        local_spans.append(pixels)
                        local_support |= supports[0] | supports[1]
                spans.append(pixels);features["gap_used"] |= bool(len(gaps))
                if len(gaps):inferred_gap[gaps[:,0],gaps[:,1]]=True
        if diagnostics:
            corridor = np.zeros(z.shape, bool)
            for pixels in spans: corridor[pixels[:, 0], pixels[:, 1]] = True
            diag["bridge_corridor_mask"] = corridor
            diag["inferred_gap_mask"] = inferred_gap
            diag["depth_rejected_mask"] = rejected_depth
        complete &= not clipped
        features.update(anchor_observable=observable, search_complete=bool(complete), work_pixels=int(kept.sum()),
                        depth_jump_threshold_m=float(depth_jump_m), depth_rejected_pixels=int(rejected_depth.sum()),
                        inferred_gap_pixels=int(inferred_gap.sum()))
        local_positive = anchor_policy == "local_observed" and bool(local_spans) and not all(observable)
        if local_positive:
            # Limit later sag/semantic inspection to supported spanning components.
            spans = local_spans
            if diagnostics:
                diag["bridge_corridor_mask"][:] = False
                for pixels in spans: diag["bridge_corridor_mask"][pixels[:, 0], pixels[:, 1]] = True
        if anchor_policy != "complete":
            features["anchor_policy"] = anchor_policy
            features["local_anchor_positive_used"] = local_positive
            features["inferred_depth_pixels"] = 0
            if diagnostics: diag["local_anchor_support_mask"] = local_support
        bridge = "yes" if spans and (all(observable) or local_positive) else "no" if complete else "unknown"
        reasons = {"bridge":"observed_work_path_reaches_both_anchors" if bridge=="yes" else
                   "no_spanning_path_in_complete_search" if bridge=="no" else "anchor_or_search_unobserved_or_clipped"}
        if local_positive:
            reasons["bridge"] = "observed_path_with_local_anchor_support"
        if bridge=="yes" and features["gap_used"]:
            reasons["bridge"]="work_path_with_explicit_small_gap_inference"
        elif bridge=="unknown" and rejected_depth.any():
            reasons["bridge"]="candidate_depth_discontinuity"
        sag = "unknown"; reasons["large_sag"] = "bridge_not_observed"
        if bridge == "yes":
            # Use only spanning components, never an unrelated object's silhouette.
            points_on_bridge = np.concatenate(spans); ys,xs = points_on_bridge.T
            lo,hi = sorted((lu,ru)); bins=max(2,int(np.ceil(span)))
            coord=(u[ys,xs]-lo)*(bins/span); keep=(coord>=0)&(coord<bins)
            top=np.full(bins,np.nan)
            for j in range(bins):
                values=v[ys[keep],xs[keep]][coord[keep].astype(int)==j]
                if values.size:top[j]=float(values.min())
            seen=np.isfinite(top);features["span_coverage"]=float(seen.mean())
            centers=lo+(np.arange(bins)+.5)*span/bins
            if diagnostics:
                diag["sag_path_xy"] = np.column_stack((centers[seen], top[seen])) @ np.stack((axis, d))
            if seen.mean() >= .95:
                line=lv+(centers-lu)*(rv-lv)/(ru-lu)
                envelope=np.interp(np.clip(u,lo,hi),centers[seen],top[seen])
                upper_search=search&(u>=lo)&(u<=hi)&(v<=envelope)
                if np.all((valid&~arms)[upper_search]):
                    ratio=float(max(0.,np.quantile(top[seen]-line[seen],.9))/span)
                    features["sag_ratio"]=ratio;sag="yes" if ratio>=self.sag_ratio else "no"
                    reasons["large_sag"]="upper_envelope_vs_sloped_anchor_chord"
                else:reasons["large_sag"]="upper_envelope_region_unobserved"
            else:reasons["large_sag"]="insufficient_observed_span_coverage"
        hanging=[]
        for i,name in enumerate(("left","right")):
            matching=[p for p,hits in components if hits[i]]
            extent=width=0.
            if matching:
                ys,xs=np.concatenate(matching).T
                extent=float(max(0.,np.quantile(v[ys,xs]-points[i][1],.95)))
                width=float(np.quantile(u[ys,xs],.95)-np.quantile(u[ys,xs],.05))
            if diagnostics:
                mask = np.zeros(z.shape, bool)
                for pixels in matching: mask[pixels[:, 0], pixels[:, 1]] = True
                diag[name+"_drape_component_mask"] = mask
            if observable[i]:
                features[name+"_hanging_extent_px"]=extent;features[name+"_hanging_width_px"]=width
            hanging.append(extent/span>=self.drape_ratio)
        if not all(observable):drape="unknown"
        elif all(hanging):drape="both"
        elif not complete:drape="unknown"
        elif hanging[0]:drape="left"
        elif hanging[1]:drape="right"
        else:drape="neither"
        reasons["drape"]="observed_anchor_connected_extent" if drape!="unknown" else "opposite_anchor_or_search_unobserved"
        features["hanging_extent_ratio_threshold"]=self.drape_ratio
        if diagnostics: diag["sag_path_used"] = sag != "unknown"
        return BridgeDrapeResult(bridge,sag,drape,reasons,features,foreground,kept,~valid,diagnostics=diag)
