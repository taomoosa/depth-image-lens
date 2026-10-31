"""Ordered visible external-contour evidence, not material-edge identity."""
from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np
from .debug import DebugResultMixin, _opencv, depth_canvas
from .geometry import CameraIntrinsics, bool_mask
from .depth import _numeric_image, _validate_depth_scale


def _canonical(points):
    p=np.asarray(points,dtype=np.int32)
    area=float(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))/2
    if area<0:p=p[::-1]
    i=int(np.lexsort((p[:,0],p[:,1]))[0])
    return np.roll(p,-i,axis=0).copy(),abs(area)


def _fit(points):
    p=np.asarray(points,float);center=p.mean(axis=0);v=p-center
    _,_,axes=np.linalg.svd(v,full_matrices=False)
    direction=axes[0]
    if np.dot(direction,p[-1]-p[0])<0:direction=-direction
    along=v@direction;distance=np.linalg.norm(v-along[:,None]*direction,axis=1)
    arc=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
    chord=float(np.linalg.norm(p[-1]-p[0]));span=float(np.ptp(along))
    edges=np.diff(p,axis=0);lengths=np.linalg.norm(edges,axis=1);directions=edges[lengths>0]
    turns=np.degrees(np.arctan2(directions[:-1,0]*directions[1:,1]-directions[:-1,1]*directions[1:,0],(directions[:-1]*directions[1:]).sum(axis=1)))
    return dict(arc_length_px=arc,chord_length_px=chord,fit_length_px=span,
                raw_signed_turn_deg=float(turns.sum()),raw_absolute_turn_deg=float(np.abs(turns).sum()),
                line_rms_px=float(np.sqrt(np.mean(distance**2))),
                line_p95_px=float(np.quantile(distance,.95)),line_max_error_px=float(distance.max()),
                arc_chord_ratio=arc/chord if chord>1e-12 else None,
                fit_endpoints_xy=[(center+along.min()*direction).tolist(),(center+along.max()*direction).tolist()])


def _straight(fit,p):
    return (fit['chord_length_px']>0 and fit['line_max_error_px']<=p['line_error_px']
            and fit['arc_chord_ratio']<=p['max_arc_chord_ratio'])


def _periodic_interp(points,s,positions,total):
    return np.stack([np.interp(np.mod(positions,total),np.r_[s,total],np.r_[points[:,d],points[0,d]]) for d in range(2)],axis=-1)


def _turns(points,s,total,p):
    # Uniform arclength samples avoid axis/diagonal sampling-density bias.
    n=max(4,int(np.ceil(total)));step=total/n;u=np.arange(n)*step
    q=_periodic_interp(points,s,u,total)
    if p['smoothing_px']>0:
        radius=int(np.ceil(3*p['smoothing_px']/step))
        offsets=np.arange(-radius,radius+1);weights=np.exp(-.5*(offsets*step/p['smoothing_px'])**2);weights/=weights.sum()
        q=sum(w*np.roll(q,int(o),axis=0) for o,w in zip(offsets,weights))
    w=p['corner_window_px']
    a=q-_periodic_interp(q,u,u-w,total);b=_periodic_interp(q,u,u+w,total)-q
    cross=a[:,0]*b[:,1]-a[:,1]*b[:,0];dot=(a*b).sum(axis=1)
    turn=np.degrees(np.arctan2(cross,dot))
    supported=(np.linalg.norm(a,axis=1)>1e-6)&(np.linalg.norm(b,axis=1)>1e-6)&(total>=4*w)
    turn[~supported]=0
    candidates=np.flatnonzero(supported&(np.abs(turn)>=p['corner_turn_deg']))
    selected=[]
    for i in sorted(candidates,key=lambda i:(-abs(turn[i]),int(i))):
        if all(min(abs(u[i]-u[j]),total-abs(u[i]-u[j]))>=p['min_corner_spacing_px'] for j in selected):selected.append(int(i))
    corners={}
    for i in selected:
        distance=np.abs(s-u[i]);distance=np.minimum(distance,total-distance);j=int(np.argmin(distance))
        if j not in corners or abs(turn[i])>abs(corners[j]):corners[j]=float(turn[i])
    # Evaluate local turn at raw-edge midpoints; the raw geometry is never replaced.
    mid=np.mod(s+np.diff(np.r_[s,total])/2,total)
    local=np.interp(mid,np.r_[u,total],np.r_[np.abs(turn),abs(turn[0])])
    return corners,local,dict(sample_positions_xy=q,sample_turn_deg=turn,sample_support=supported)


def _ranges(n,cuts):
    cuts=sorted(set(int(i)%n for i in cuts)) or [0]
    return [(a,(b-a)%n or n) for a,b in zip(cuts,cuts[1:]+cuts[:1])]


@dataclass(frozen=True)
class OuterContourResult(DebugResultMixin):
    contours: list
    holes: list
    parameters: dict
    summary: dict
    object_mask: np.ndarray=field(repr=False)
    observed_mask: np.ndarray=field(repr=False)
    diagnostics: dict=field(default_factory=dict,repr=False)

    def to_dict(self):
        return dict(contours=self.contours,holes=self.holes,parameters=self.parameters,summary=self.summary)

    @staticmethod
    def _id_map(records,label):
        mapping={}
        for record in records:
            key=record.get('id')
            if isinstance(key,(bool,np.bool_)) or not isinstance(key,(int,np.integer)) or key in mapping:
                raise ValueError(f'{label} requires unique integer IDs')
            mapping[int(key)]=record
        return mapping

    def largest_contour(self,*,by='area_px2'):
        """Return a detached largest-contour record; ties use the smallest ID.

        Polygon area includes holes. foreground_pixel_count instead measures the
        observed connected foreground, excluding holes and separate nested islands.
        Empty results return None. This does not alter classification or queries.
        """
        if by not in ('area_px2','foreground_pixel_count','perimeter_px'):
            raise ValueError('by must be area_px2, foreground_pixel_count or perimeter_px')
        contours=self._id_map(self.contours,'contours')
        for contour in contours.values():
            value=contour.get(by)
            if not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value) or value<0:
                raise ValueError(f'contour has no valid {by} measurement')
        if not contours:return None
        selected=min(contours.values(),key=lambda c:(-c[by],int(c['id'])))
        return deepcopy(selected)

    def component_mask(self, contour_id=None):
        """Return a new bool mask for a visible 8-connected OuterContour component.

        None selects largest polygon area (the result's documented tie policy).
        No contours returns an all-False mask; an unknown explicit ID raises
        ValueError. A missing, malformed or out-of-bounds start point also raises
        ValueError. IDs are looked up, never interpreted as component-label indices.
        Re-label object_mask & observed_mask, using a raw contour point as seed.
        Holes stay False and separate objects nested inside holes stay excluded.
        Uses optional OpenCV, as does OuterContourDetector. Touching objects already
        in one 8-connected component cannot be separated by this helper.
        """
        cv2 = _opencv()
        if contour_id is None:
            contour = self.largest_contour()
            if contour is None:
                return np.zeros_like(self.object_mask, dtype=bool)
        else:
            if isinstance(contour_id, (bool, np.bool_)) or not isinstance(contour_id, (int, np.integer)):
                raise ValueError('contour_id must be an integer ID')
            matches = [c for c in self.contours if c['id'] == contour_id]
            if len(matches) != 1:
                raise ValueError('contour_id must identify exactly one contour')
            contour = matches[0]
        visible = self.object_mask & self.observed_mask
        try:
            seed = contour['points_xy'][0]
            if len(seed) != 2:
                raise ValueError('contour start must have exactly two coordinates')
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError('contour must contain a start point in points_xy') from error
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               for v in seed):
            raise ValueError('contour start coordinates must be finite integers')
        x0, y0 = map(int, seed)
        if not (0 <= x0 < visible.shape[1] and 0 <= y0 < visible.shape[0]):
            raise ValueError('contour start is outside the mask grid')
        _, labels = cv2.connectedComponents(visible.astype(np.uint8), connectivity=8)
        label = labels[y0, x0]
        if label == 0:
            raise ValueError('contour point no longer belongs to visible foreground')
        return labels == label

    def resolve_sequence(self,contour_id=None):
        """Return detached type/id/record entries in the stored cyclic order.

        None selects the largest polygon area; an empty result returns []. An
        explicit missing ID or invalid/ambiguous sequence raises ValueError.
        IDs are looked up, never interpreted as list indices. No graph is rebuilt.
        """
        contours=self._id_map(self.contours,'contours')
        if contour_id is None:
            largest=self.largest_contour()
            if largest is None:return []
            contour_id=largest['id']
        if isinstance(contour_id,(bool,np.bool_)) or not isinstance(contour_id,(int,np.integer)) or contour_id not in contours:
            raise ValueError('contour_id must identify an existing contour')
        contour=contours[contour_id]
        lookup={'segment':self._id_map(contour['segments'],'segments'),
                'corner':self._id_map(contour['corners'],'corners')}
        resolved=[];seen=set()
        for entry in contour['sequence']:
            kind=entry.get('type');key=entry.get('id')
            if kind not in lookup or isinstance(key,(bool,np.bool_)) or not isinstance(key,(int,np.integer)) or key not in lookup[kind]:
                raise ValueError('sequence references an unknown type or record ID')
            identity=(kind,int(key))
            if identity in seen:raise ValueError('sequence contains a duplicate record reference')
            seen.add(identity);resolved.append(dict(type=kind,id=int(key),record=deepcopy(lookup[kind][key])))
        if seen!={(kind,key) for kind,records in lookup.items() for key in records}:
            raise ValueError('sequence omits a segment or corner record')
        return resolved

    def query_edge_length(self,target_length,*,unit='px',tolerance=None,mode='match',
                          min_support_px=20.,max_line_error_px=None,max_metric_line_error_m=.005,require_corner_endpoints=False):
        """Find visible straight interval candidates, never a particular material edge.

        Length uses the projected fitted span. mode='at_least' permits longer
        intervals; 'match' compares complete returned interval length. Missing
        metric samples/clipped intervals stay unknown, not nonmatches.
        """
        if tolerance is None:tolerance=5. if unit=='px' else .005
        if (not np.isfinite([target_length,tolerance,min_support_px]).all() or target_length<=0
                or min(tolerance,min_support_px)<0 or unit not in ('px','m') or mode not in ('match','at_least')):
            raise ValueError('positive target, nonnegative tolerance/support, unit px/m and mode match/at_least required')
        if max_line_error_px is None:max_line_error_px=self.parameters['line_error_px']
        if not np.isfinite([max_line_error_px,max_metric_line_error_m]).all() or min(max_line_error_px,max_metric_line_error_m)<0 or type(require_corner_endpoints) is not bool:
            raise ValueError('finite nonnegative line error and bool corner requirement required')
        rows=[]
        for contour in self.contours:
            for seg in contour['segments']:
                reasons=[]
                if seg['kind']!='straight':continue
                if seg['visibility']!='observed':reasons.append('clipped_or_unobserved_boundary')
                if contour['status']!='computed':reasons.append('ambiguous_or_insufficient_contour')
                if seg['arc_length_px']<min_support_px:reasons.append('insufficient_visible_support')
                if seg['line_max_error_px']>max_line_error_px:reasons.append('line_fit_exceeds_query_limit')
                if require_corner_endpoints and any(x is None for x in seg['corner_ids']):reasons.append('two_supported_corner_endpoints_required')
                length=seg['fit_length_px'] if unit=='px' else seg['metric'].get('fit_length_m')
                if unit=='m' and seg['metric']['status']!='measured':reasons.append('metric_length_unavailable')
                elif unit=='m' and seg['metric']['line_max_error_m']>max_metric_line_error_m:reasons.append('nonstraight_3d_interval')
                epsilon=0. if length is None else 8*np.finfo(float).eps*max(1.,abs(length),abs(target_length),abs(tolerance))
                matching=None if length is None else (abs(length-target_length)<=tolerance+epsilon if mode=='match' else length+tolerance+epsilon>=target_length)
                status='unknown' if reasons else 'candidate' if matching else 'not_matched'
                rows.append(dict(contour_id=contour['id'],segment_id=seg['id'],status=status,length=length,unit=unit,reasons=reasons))
        candidates=[r for r in rows if r['status']=='candidate']
        incomplete=any(c['status']!='computed' or c['clipped'] or any(s['kind']=='unknown' for s in c['segments']) for c in self.contours)
        state='candidate_found' if candidates else 'unknown' if incomplete or any(r['status']=='unknown' for r in rows) or not self.contours else 'no_matching_visible_interval'
        return dict(status=state,target_length=float(target_length),unit=unit,length_kind='tls_projected_span',tolerance=float(tolerance),mode=mode,
                    min_support_px=float(min_support_px),max_line_error_px=float(max_line_error_px),max_metric_line_error_m=float(max_metric_line_error_m),
                    require_corner_endpoints=require_corner_endpoints,intervals=rows,candidate_count=len(candidates),
                    material_edge_identity='unknown',scope='visible_straight_interval_not_original_material_edge_or_grasp_proof')

    def render_debug(self,color=None,*,color_order='RGB',depth=None,depth_scale_m=.001,
                     show_raw=True,show_ids=True,panel=True):
        cv=_opencv();shape=self.object_mask.shape
        if color_order not in ('RGB','BGR') or any(type(v) is not bool for v in (show_raw,show_ids,panel)):
            raise ValueError('RGB/BGR and boolean rendering switches required')
        if color is not None:
            a=np.asarray(color)
            if a.dtype!=np.uint8 or a.shape!=(*shape,3):raise ValueError('color must match uint8 HxWx3 mask grid')
            canvas=(a if color_order=='RGB' else a[...,::-1]).copy()
        elif depth is not None:
            canvas=depth_canvas(depth,depth_scale_m)
            if canvas.shape[:2]!=shape:raise ValueError('depth/mask shape mismatch')
        else:canvas=np.where(self.object_mask[...,None],80,25).astype(np.uint8).repeat(3,axis=2)
        for h in self.holes:cv.polylines(canvas,[np.asarray(h['points_xy'],np.int32)],True,(140,140,140),1)
        for c in self.contours:
            if show_raw:cv.polylines(canvas,[np.asarray(c['points_xy'],np.int32)],True,(235,235,235),1)
            for seg in c['segments']:
                ink=(45,215,85) if seg['kind']=='straight' else (70,150,250) if seg['kind']=='curve' else (180,180,180)
                if seg['visibility']!='observed':ink=(230,100,210)
                q=np.asarray(seg['points_xy'],np.int32);cv.polylines(canvas,[q],False,ink,2)
                if show_ids:
                    xy=tuple(q[len(q)//2]);cv.putText(canvas,f"{c['id']}:{seg['id']}",xy,cv.FONT_HERSHEY_SIMPLEX,.35,(255,230,90),1)
            for corner in c['corners']:
                xy=tuple(corner['xy']);ink=(255,180,35) if corner['visibility']=='observed' else (230,100,210)
                cv.circle(canvas,xy,3,ink,-1)
                if show_ids:cv.putText(canvas,f"c{corner['id']} {corner['turn_deg']:+.0f}deg",(xy[0]+4,xy[1]-4),cv.FONT_HERSHEY_SIMPLEX,.32,ink,1)
            if show_ids:cv.circle(canvas,tuple(c['points_xy'][c['sequence_start_index']]),5,(255,255,255),1)
        if panel:
            lines=['OuterContour: observed mask geometry, not material-edge identity',
                   'Green straight / blue curve / gray short / magenta clipped-unknown',
                   'Orange corners / gray holes; white ring = cyclic sequence start',
                   f"Contours {len(self.contours)}; line error {self.parameters['line_error_px']} px; turn {self.parameters['corner_turn_deg']} deg",
                   f"Smoothing {self.parameters['smoothing_px']} px; window {self.parameters['corner_window_px']} px",
                   'Raw pixel-center lengths differ from physical material dimensions.']
            pad=np.full((max(shape[0],145),max(shape[1],650),3),24,np.uint8);pad[:shape[0],:shape[1]]=canvas
            footer=np.full((145,pad.shape[1],3),24,np.uint8)
            for i,line in enumerate(lines):cv.putText(footer,line,(8,22+21*i),cv.FONT_HERSHEY_SIMPLEX,.4,(235,235,235),1,cv.LINE_AA)
            canvas=np.concatenate((pad,footer))
        return canvas if color_order=='RGB' else np.ascontiguousarray(canvas[...,::-1])


class OuterContourDetector:
    """Closed exterior decomposition; optional depth never changes pixel geometry."""
    def __init__(self,*,smoothing_px=1.,corner_window_px=8.,corner_turn_deg=45.,
                 min_corner_spacing_px=8.,straight_turn_deg=5.,line_error_px=1.5,
                 max_arc_chord_ratio=1.15,min_section_length_px=8.,min_straight_length_px=20.,clip_margin_px=1,
                 max_depth_jump_m=.02):
        numeric=[smoothing_px,corner_window_px,corner_turn_deg,min_corner_spacing_px,straight_turn_deg,
                 line_error_px,max_arc_chord_ratio,min_section_length_px,min_straight_length_px,max_depth_jump_m]
        if not np.isfinite(numeric).all():raise ValueError('finite contour parameters required')
        if not 0<=smoothing_px<=20 or not 1<=corner_window_px<=100 or not 0<corner_turn_deg<=180:
            raise ValueError('smoothing 0..20 px, window 1..100 px and turn (0,180] deg required')
        if not 1<=min_corner_spacing_px<=200 or not 0<=straight_turn_deg<=180 or line_error_px<0 or max_arc_chord_ratio<1 or min_section_length_px<0 or min_straight_length_px<0 or max_depth_jump_m<=0:
            raise ValueError('invalid spacing, straightness, support or depth-jump setting')
        if type(clip_margin_px) is not int or not 0<=clip_margin_px<=10:raise ValueError('clip_margin_px integer 0..10 required')
        self.parameters=dict(zip(['smoothing_px','corner_window_px','corner_turn_deg','min_corner_spacing_px','straight_turn_deg','line_error_px','max_arc_chord_ratio','min_section_length_px','min_straight_length_px','max_depth_jump_m'],map(float,numeric)))
        self.parameters['clip_margin_px']=clip_margin_px

    def detect(self,object_mask,*,observed_mask=None,depth=None,depth_scale_m=.001,
               intrinsics=None,registered_to_mask=False,depth_valid_mask=None,diagnostics=False):
        cv=_opencv();a=np.asarray(object_mask)
        if a.dtype!=bool or a.ndim!=2 or min(a.shape)<2:raise ValueError('object_mask must be bool HxW')
        if type(diagnostics) is not bool:raise ValueError('diagnostics must be bool')
        obj=a.copy();shape=obj.shape
        observed=np.ones(shape,bool) if observed_mask is None else bool_mask(observed_mask,shape,'observed_mask').copy()
        visible=obj&observed
        # Domain edges are unknown even if the mask makes a crisp rectangle there.
        margin=self.parameters['clip_margin_px'];radius=margin+1
        safe=cv.erode(observed.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8),borderType=cv.BORDER_CONSTANT,borderValue=0)>0
        z=valid=None;metric_reason='depth_not_supplied'
        if depth is not None:
            try:
                _validate_depth_scale(depth_scale_m)
                if registered_to_mask is not True:raise ValueError('registered_to_mask=True required')
                if not isinstance(intrinsics,CameraIntrinsics):raise ValueError('CameraIntrinsics required')
                with np.errstate(over='ignore',invalid='ignore'):z=_numeric_image(depth,'depth')*depth_scale_m
                if z.shape!=shape:raise ValueError('depth/mask shape mismatch')
                valid=np.isfinite(z)&(z>0)&observed
                if depth_valid_mask is not None:valid &= bool_mask(depth_valid_mask,shape,'depth_valid_mask')
                metric_reason=None
            except (ValueError,TypeError,OverflowError) as error:
                z=valid=None;metric_reason='invalid_optional_metric_input: '+str(error)
        raw,hierarchy=cv.findContours(visible.astype(np.uint8),cv.RETR_CCOMP,cv.CHAIN_APPROX_NONE)
        hierarchy=[] if hierarchy is None else hierarchy[0]
        roots=sorted([i for i in range(len(raw)) if hierarchy[i][3]<0],key=lambda i:(int(raw[i][:,0,1].min()),int(raw[i][:,0,0].min()),i))
        contours=[];holes=[];debug={}
        _,component_labels,component_stats,_=cv.connectedComponentsWithStats(visible.astype(np.uint8),connectivity=8)
        for cid,idx in enumerate(roots):
            pts,area=_canonical(raw[idx][:,0,:]);n=len(pts)
            for j in range(len(raw)):
                if hierarchy[j][3]==idx:
                    hp,ha=_canonical(raw[j][:,0,:]);holes.append(dict(id=len(holes),parent_contour_id=cid,points_xy=hp.tolist(),area_px=ha,area_px2=ha,perimeter_px=float(np.linalg.norm(np.roll(hp,-1,axis=0)-hp,axis=1).sum()),role='hole_not_external_edge'))
            record=self._decompose(pts,cid,safe,z,valid,intrinsics,metric_reason,diagnostics,debug)
            record['area_px']=area
            label=component_labels[pts[0,1],pts[0,0]]
            record.update(area_px2=area,hole_area_px2=float(sum(h['area_px2'] for h in holes if h['parent_contour_id']==cid)),
                          foreground_pixel_count=int(component_stats[label,cv.CC_STAT_AREA]),perimeter_px=record['arc_length_px'])
            if area==0:record['orientation']='degenerate_zero_area'
            record['hole_ids']=[h['id'] for h in holes if h['parent_contour_id']==cid];contours.append(record)
        return OuterContourResult(contours,holes,self.parameters.copy(),dict(status='computed' if contours else 'no_visible_contour',
            coordinate_order='xy; arrays yx',orientation='clockwise_in_image_coordinates',closure='cyclic_raw_pixel_center_edges',
            metric_input_reason=metric_reason,component_count=len(contours),hole_count=len(holes),
            warnings=['supplied_mask_boundary_not_verified_material_edge','holes_are_not_external_contours','boundary_depth_can_be_mixed_even_when_finite'],
            rejected_optional_metric=depth is not None and metric_reason is not None),obj,observed,debug)

    def _decompose(self,pts,cid,safe,z,valid,intrinsics,metric_reason,diagnostics,debug):
        p=self.parameters;n=len(pts);steps=np.linalg.norm(np.roll(pts,-1,axis=0)-pts,axis=1)
        s=np.r_[0,np.cumsum(steps[:-1])];total=float(steps.sum())
        good=safe[pts[:,1],pts[:,0]];edge_good=good&np.roll(good,-1)
        repeated=len(np.unique(pts,axis=0))<n
        status='ambiguous_self_touch' if repeated else 'insufficient_support' if n<8 or total<4*p['corner_window_px'] else 'computed'
        corner_map={};local=np.zeros(n);diag={}
        if n>20000:status='insufficient_computation_budget'
        elif n>=8 and total>=4*p['corner_window_px']:corner_map,local,diag=_turns(pts,s,total,p)
        corners=[dict(id=i,point_index=int(j),xy=pts[j].tolist(),turn_deg=t,
                      kind='convex' if t>0 else 'concave',visibility='observed' if good[j] and status=='computed' else 'uncertain')
                 for i,(j,t) in enumerate(sorted(corner_map.items()))]
        corner_ids={c['point_index']:c['id'] for c in corners}
        cuts=set(corner_map)|set(np.flatnonzero(edge_good!=np.roll(edge_good,1)).tolist())
        groups=[]
        for start,count in _ranges(n,cuts):
            indices=(start+np.arange(count+1))%n;q=pts[indices]
            fit=_fit(q)
            if status!='computed':groups.append((start,count,'unknown'));continue
            if _straight(fit,p):
                groups.append((start,count,'straight' if fit['arc_length_px']>=p['min_straight_length_px'] else 'unknown'));continue
            labels=local[indices[:-1]]<=p['straight_turn_deg']
            splits=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1,count]
            for a,b in zip(splits[:-1],splits[1:]):
                kind='straight' if labels[a] else 'curve';groups.append(((start+int(a))%n,int(b-a),kind))
        # Only merge adjacent runs across the canonical seam when no true cut separates them.
        if len(groups)>1 and groups[-1][2]==groups[0][2] and groups[0][0] not in cuts:
            a,b=groups[-1],groups[0];groups=[(a[0],a[1]+b[1],a[2])]+groups[1:-1]
        # A brief low-turn patch on a curve is not a supported straight section.
        classified=[]
        for start,count,kind in groups:
            fit=_fit(pts[(start+np.arange(count+1))%n])
            if kind=='straight' and (not _straight(fit,p) or fit['arc_length_px']<p['min_straight_length_px']):kind='curve'
            if classified and classified[-1][2]==kind and start not in cuts:
                prev=classified[-1];classified[-1]=(prev[0],prev[1]+count,kind)
            else:classified.append((start,count,kind))
        if len(classified)>1 and classified[-1][2]==classified[0][2] and classified[0][0] not in cuts:
            a,b=classified[-1],classified[0];classified=[(a[0],a[1]+b[1],a[2])]+classified[1:-1]
        groups=classified
        segments=[]
        for start,count,kind in groups:
            indices=(start+np.arange(count+1))%n;q=pts[indices];fit=_fit(q)
            if kind=='straight' and not _straight(fit,p):kind='curve'
            if fit['arc_length_px']<p['min_section_length_px'] or status!='computed':kind='unknown'
            observed=bool(edge_good[indices[:-1]].all());metric=dict(status='unknown',reason=metric_reason or 'incomplete_or_discontinuous_depth',valid_fraction=0.,arc_length_m=None,chord_length_m=None,fit_length_m=None,observed_adjacent_length_m=None,sample_valid=None,adjacent_pair_valid=None,fit_endpoints_xyz_m=None,line_rms_m=None,line_max_error_m=None)
            if z is not None:
                zz=z[q[:,1],q[:,0]];ok=valid[q[:,1],q[:,0]]
                with np.errstate(over='ignore',invalid='ignore'):
                    xyz=np.c_[(q[:,0]-intrinsics.cx)*zz/intrinsics.fx,(q[:,1]-intrinsics.cy)*zz/intrinsics.fy,zz]
                ok=ok&np.isfinite(xyz).all(axis=1);pairs=ok[:-1]&ok[1:]&(np.abs(np.diff(zz))<=p['max_depth_jump_m'])
                distances=np.linalg.norm(np.diff(xyz,axis=0),axis=1);pairs &= np.isfinite(distances)
                metric.update(valid_fraction=float(ok.mean()),observed_adjacent_length_m=float(distances[pairs].sum()),sample_valid=ok.tolist(),adjacent_pair_valid=pairs.tolist())
                if ok.all() and pairs.all():
                    v=xyz-xyz.mean(axis=0);_,_,axes=np.linalg.svd(v,full_matrices=False)
                    perpendicular=np.linalg.norm(v-(v@axes[0])[:,None]*axes[0],axis=1)
                    metric.update(line_rms_m=float(np.sqrt(np.mean(perpendicular**2))),line_max_error_m=float(perpendicular.max()),status='measured',reason='registered_boundary_depth_not_verified_material_size',arc_length_m=float(distances.sum()),chord_length_m=float(np.linalg.norm(xyz[-1]-xyz[0])),fit_length_m=float(np.ptp(v@axes[0])),fit_endpoints_xyz_m=[(xyz.mean(axis=0)+(v@axes[0]).min()*axes[0]).tolist(),(xyz.mean(axis=0)+(v@axes[0]).max()*axes[0]).tolist()])
            segments.append(dict(id=len(segments),kind=kind,start_index=int(start),end_index=int((start+count)%n),edge_count=int(count),
                                 points_xy=q.tolist(),visibility='observed' if observed else 'uncertain',
                                 reasons=[] if observed else ['frame_or_observation_domain_boundary'],
                                 corner_ids=[corner_ids.get(start),corner_ids.get((start+count)%n)],
                                 mean_abs_turn_deg=float(local[indices[:-1]].mean()),max_abs_turn_deg=float(local[indices[:-1]].max()),metric=metric,**fit))
        sequence=[]
        for seg in segments:
            if seg['corner_ids'][0] is not None:sequence.append(dict(type='corner',id=seg['corner_ids'][0]))
            sequence.append(dict(type='segment',id=seg['id']))
        if diagnostics:debug[str(cid)]=diag
        return dict(id=cid,status=status,points_xy=pts.tolist(),closed=True,orientation='clockwise_in_image_coordinates',
                    edge_count=n,arc_length_px=total,clipped=bool(not edge_good.all()),self_touch=repeated,
                    segments=segments,corners=corners,sequence=sequence,sequence_start_index=segments[0]['start_index'] if segments else 0,
                    coverage_edge_count=sum(x['edge_count'] for x in segments))
