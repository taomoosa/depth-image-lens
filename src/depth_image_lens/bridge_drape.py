"""NumPy-only, static background-difference shape heuristics. Not grasp proof."""
from collections import deque
from dataclasses import dataclass, field, replace
import numpy as np

__version__ = "0.2.0"


def _mask(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype != bool:
        raise ValueError(f"{name} must be a bool array with shape {shape}")
    return a



def _depth_arrays(depth, difference, depth_unit, difference_unit, difference_sign, background_valid, observed_mask):
    z = np.asarray(depth, dtype=float); diff = np.asarray(difference, dtype=float)
    if z.ndim != 2 or z.shape != diff.shape or min(z.shape) < 2:
        raise ValueError("depth/difference must be same HxW shape")
    if depth_unit not in ("m", "mm") or difference_unit != depth_unit:
        raise ValueError("declare the same m/mm unit for depth and difference")
    if difference_sign not in ("background-current", "current-background"):
        raise ValueError("unknown difference_sign")
    scale = .001 if depth_unit == "mm" else 1.
    z = z*scale; diff = diff*scale*(1 if difference_sign == "background-current" else -1)
    with np.errstate(invalid="ignore", over="ignore"):
        background = z + diff
    valid = np.isfinite(z) & (z > 0) & np.isfinite(diff) & np.isfinite(background) & (background > 0)
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

    def to_dict(self, *, include_masks=False):
        """Finite JSON scalars by default; optional bool-list masks can be large."""
        result = {k: getattr(self, k) for k in
                  ("bridge", "large_sag", "drape", "reasons", "features", "validation_status", "scope")}
        if self.auto: result["auto"] = self.auto
        if include_masks:
            result.update({k: getattr(self, k).tolist() for k in
                           ("foreground_mask", "work_mask", "invalid_mask")})
        return result


class BridgeDrapeDetector:
    """Manual ROIs or bridge-first shape candidates, not semantic hand detection.

    Depth and difference must share pixels, units, depth meaning and valid-data
    conventions. Difference=background-current is positive toward the camera.
    Invalid background must be NaN in difference or false in background_valid.
    Zero-filled/clipped differences cannot recover missing background provenance.
    """
    def __init__(self, *, threshold_m=.02, min_area=20, anchor_band_px=3,
                 max_gap_px=0, sag_ratio=.15, drape_ratio=.25):
        if not np.isfinite([threshold_m, sag_ratio, drape_ratio]).all() or min(threshold_m, sag_ratio, drape_ratio) <= 0:
            raise ValueError("threshold and ratios must be positive finite values")
        if any(type(x) is not int for x in (min_area, anchor_band_px, max_gap_px)) or min_area < 1 or anchor_band_px < 1 or not 0 <= max_gap_px <= 2:
            raise ValueError("positive integer area/band and gap0..2 pixels required")
        self.threshold_m = float(threshold_m)
        self.min_area = min_area
        self.anchor_band_px = anchor_band_px
        self.max_gap_px = max_gap_px
        self.sag_ratio = float(sag_ratio)
        self.drape_ratio = float(drape_ratio)

    def _components(self, foreground, traversable):
        """Eight-neighbour paths; optional tiny gaps only through observed pixels.

        No new foreground is created. Invalid/excluded pixels are never jumped.
        """
        remaining = foreground.copy(); h, w = remaining.shape
        for sy, sx in zip(*np.nonzero(foreground)):
            if not remaining[sy, sx]:
                continue
            queue = deque([(int(sy), int(sx))]); remaining[sy, sx] = False
            pixels = []; used_gap = False
            while queue:
                y, x = queue.popleft(); pixels.append((y, x))
                for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                    for step in range(1, self.max_gap_px + 2):
                        yy, xx = y + dy*step, x + dx*step
                        if not (0 <= yy < h and 0 <= xx < w) or not traversable[yy, xx]:
                            break
                        if foreground[yy, xx]:
                            if remaining[yy, xx]:
                                remaining[yy, xx] = False; queue.append((yy, xx))
                                used_gap |= step > 1
                            break
            if len(pixels) >= self.min_area:
                yield np.asarray(pixels, dtype=int), used_gap

    def detect_auto(self, depth, difference, *, upper_band=None, min_span_px=30,
                    max_height_delta_px=12, depth_jump_m=.10, down=(0,1), **options):
        """Bridge first, then drape necks. All ROIs are unverified image candidates.

        upper_band=(v_min,v_max) is along normalized image down. Its lower edge
        separates upper arm clutter from the deformable-object region. Default: top quarter
        of the image. A poor boundary can miss the deformable object; this is not semantic detection.
        options accepts detect() units, validity masks, arm exclusion and work_roi, but not hand ROIs
        or visibility declarations. Missing opposite evidence stays unknown.
        """
        allowed={"depth_unit","difference_unit","difference_sign","background_valid","observed_mask","arm_mask","work_roi"}
        if set(options)-allowed:raise ValueError("unsupported auto option; manual ROIs use detect()")
        if not np.isfinite([min_span_px,max_height_delta_px,depth_jump_m]).all() or min(min_span_px,max_height_delta_px,depth_jump_m)<=0:
            raise ValueError("positive finite auto thresholds required")
        z,diff,valid=_depth_arrays(depth,difference,options.get("depth_unit","m"),options.get("difference_unit","m"),
            options.get("difference_sign","background-current"),options.get("background_valid"),options.get("observed_mask"))
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
        fg=valid&(diff>self.threshold_m);usable=valid&~arm&roi
        work=fg&usable&(v>band[1])
        meta={"stage":"search","identity":"image transverse left/right, not robot IDs","upper_band":band.tolist(),
              "hand_roi_candidates":{},"opposite_search_region":None,"physical_contact_verified":False,
              "candidate_knowledge":"shape and same-height foreground only; no semantic hand model"}
        def uncertain(reason, bridge="unknown", drape="unknown"):
            return BridgeDrapeResult(bridge,"unknown",drape,dict.fromkeys(("bridge","large_sag","drape"),reason),
                {"sag_ratio":None,"span_coverage":None,"valid_fraction":float(valid.mean()),**meta.get('observed_features',{})},fg,work,~valid,auto=meta)
        # An upper cut cannot be repaired by assuming invisible hand positions.
        if np.any(fg&roi&~arm&(v<=v.min()+1)):
            meta['stage']='upper_clipped';return uncertain('upper_foreground_clipped')
        parts=[p for p,_ in self._components(work,usable&(v>band[1]))]
        parts.sort(key=lambda p:len(p),reverse=True);parts=parts[:4]
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
            r=self.detect(depth,difference,boxes,down=down,**options)
            local={}
            for name,pt in [('left',a),('right',b)]:
                region=work&(np.abs(u-pt[0])<=.25*(b[0]-a[0]))
                local['image_'+name+'_local_hanging_extent_px']=float(max(0.,np.quantile(v[region]-pt[1],.95))) if region.any() else 0.
            return replace(r,features={**r.features,**local},auto=meta.copy())
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
        for _,a,b,p in sorted(bridge_candidates,key=lambda x:-x[0])[:2]:
            ys,xs=p.T;component=np.zeros(z.shape,bool);component[ys,xs]=True
            discontinuity=False
            for dy,dx in ((1,0),(0,1)):
                pair=component[:z.shape[0]-dy or None,:z.shape[1]-dx or None]&component[dy:,dx:]
                dz=np.abs(z[:z.shape[0]-dy or None,:z.shape[1]-dx or None]-z[dy:,dx:])
                discontinuity |= bool(np.any(dz[pair]>depth_jump_m))
            if discontinuity:
                meta['stage']='depth_discontinuous';return uncertain('bridge_depth_discontinuity')
            r=run_pair(a,b,'bridge_first')
            if r.bridge=='yes':return r
        if height_rejected and not bridge_candidates:
            meta['stage']='height_assumption_violation';return uncertain('upper_extrema_not_at_similar_height')
        necks=[]
        for p in parts:
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
            others=[p for p,_ in self._components(opposite,usable)]
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
               difference_sign="background-current", background_valid=None, observed_mask=None,
               arm_mask=None, work_roi=None, hand_visibility=("visible", "visible"), down=(0,1)):
        """ROIs: {'left':(x0,y0,x1,y1), 'right':(...)} with exclusive x1/y1.

        Use coarse hand boxes whose down-side edges are near the material contact.
        A thin lower band is retained as anchor; hand interiors are excluded.
        Visibility is a caller declaration ('visible', 'partial', 'hidden'), not
        inferred from valid depth. Mark occlusion in observed_mask as well.
        Units m/mm must agree. Threshold always uses metres. down is image-space.
        """
        z, diff, valid = _depth_arrays(depth, difference, depth_unit, difference_unit, difference_sign, background_valid, observed_mask)
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
        foreground = valid & (diff > self.threshold_m)
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
        features = {"sag_ratio":None, "span_coverage":None, "span_px":float(span),
                    "left_hanging_extent_px":None, "right_hanging_extent_px":None,
                    "left_hanging_width_px":None, "right_hanging_width_px":None,
                    "valid_fraction":float(valid.mean()), "foreground_pixels":int(foreground.sum()),
                    "gap_tolerance_px":self.max_gap_px, "gap_used":False}
        if span < 8 or np.any(anchors[0] & anchors[1]):
            return BridgeDrapeResult("unknown","unknown","unknown",dict.fromkeys(("bridge","large_sag","drape"),"anchors_overlap_or_near_vertical"),features,foreground,empty,~valid)
        baseline = lv + (u-lu)*(rv-lv)/(ru-lu)
        margin = span*.25
        search = roi & (u >= min(lu,ru)-margin) & (u <= max(lu,ru)+margin) & (v >= baseline-self.anchor_band_px)
        search &= ~hand_interiors
        usable = search & valid & ~arms
        work = foreground & usable
        observable = [bool(inside[i] and hand_visibility[i] == "visible" and anchors[i].any()
                      and np.all((valid & ~arms & roi)[anchors[i]])) for i in range(2)]
        complete = all(observable) and bool(np.all((valid & ~arms)[search]))
        components = []; kept = np.zeros(z.shape, bool); spans = []; clipped = False
        for pixels,gap in self._components(work, usable):
            ys,xs = pixels.T; kept[ys,xs] = True
            hits = [int(anchors[i][ys,xs].sum()) >= 3 for i in range(2)]
            if not any(hits):continue
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
            if all(hits):spans.append(pixels);features["gap_used"] |= gap
        complete &= not clipped
        features.update(anchor_observable=observable, search_complete=bool(complete), work_pixels=int(kept.sum()))
        bridge = "yes" if spans and all(observable) else "no" if complete else "unknown"
        reasons = {"bridge":"observed_work_path_reaches_both_anchors" if bridge=="yes" else
                   "no_spanning_path_in_complete_search" if bridge=="no" else "anchor_or_search_unobserved_or_clipped"}
        sag = "unknown"; reasons["large_sag"] = "bridge_not_observed"
        if bridge == "yes":
            # Use only spanning components, never an unrelated object's silhouette.
            points_on_bridge = np.concatenate(spans); ys,xs = points_on_bridge.T
            lo,hi = sorted((lu,ru)); bins=max(2,int(np.ceil(span)))
            coord=(u[ys,xs]-lo)/span*bins; keep=(coord>=0)&(coord<bins)
            top=np.full(bins,np.nan)
            for j in range(bins):
                values=v[ys[keep],xs[keep]][coord[keep].astype(int)==j]
                if values.size:top[j]=float(values.min())
            seen=np.isfinite(top);features["span_coverage"]=float(seen.mean())
            if seen.mean() >= .95:
                centers=lo+(np.arange(bins)+.5)*span/bins
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
        return BridgeDrapeResult(bridge,sag,drape,reasons,features,foreground,kept,~valid)
