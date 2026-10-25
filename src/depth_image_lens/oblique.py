"""Independent visible-shape measurements and static support hypotheses.

No hand identity, grasp proof, hole filling, or metric support reconstruction.
"""
from collections import deque
import numpy as np


def _stage(status, **values):
    return {'status': status, **values}


def _patch_centers(mask):
    h, w = mask.shape; pad = np.pad(mask, 1); centers = mask.copy()
    for dy in range(3):
        for dx in range(3): centers &= pad[dy:dy+h, dx:dx+w]
    return centers


def _graph_components(mask, z, limit):
    """Cut individual unsafe edges, retaining all alternative observed paths."""
    h, w = mask.shape; remaining = mask.copy(); parts = []; rejected = np.zeros_like(mask)
    offsets = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
    edges = []; rejected_count = 0
    for dy, dx in offsets:
        allowed = np.zeros_like(mask)
        ya,yb=max(0,-dy),min(h,h-dy);xa,xb=max(0,-dx),min(w,w-dx)
        a=np.s_[ya:yb,xa:xb]; b=np.s_[ya+dy:yb+dy,xa+dx:xb+dx]
        pair=mask[a]&mask[b]
        tolerance=4*np.finfo(float).eps*np.maximum(np.maximum(np.abs(z[a]),np.abs(z[b])),limit)
        with np.errstate(invalid='ignore'): good=np.abs(z[a]-z[b])<=limit+tolerance
        allowed[a]=pair&good
        bad=pair&~good;rejected[a]|=bad;rejected_count+=int(bad.sum())
        edges.append(allowed)
    for sy,sx in zip(*np.nonzero(mask)):
        if not remaining[sy,sx]:continue
        queue=deque([(int(sy),int(sx))]);remaining[sy,sx]=False;points=[]
        while queue:
            y,x=queue.popleft();points.append((y,x))
            for (dy,dx),edge in zip(offsets,edges):
                if edge[y,x] and remaining[y+dy,x+dx]:
                    remaining[y+dy,x+dx]=False;queue.append((y+dy,x+dx))
        parts.append(np.asarray(points,dtype=int))
    return sorted(parts,key=len,reverse=True),rejected,rejected_count//2


def _envelope(mask,u,v):
    ys,xs=np.nonzero(mask)
    if not len(xs):return np.array([]),np.array([])
    bins=np.floor(u[ys,xs]).astype(int);ids=np.unique(bins)
    top=np.full(len(ids),np.inf);np.minimum.at(top,np.searchsorted(ids,bins),v[ys,xs])
    return ids.astype(float)+.5,top


class ObliqueShapeAnalyzer:
    """Small feature pipeline; each stage declares its own inputs and assumptions."""
    def analyze(self,z,candidate_mask,valid,*,arm_mask,work_roi,hand_rois,down,
                hand_visibility,anchor_band_px,min_area,depth_jump_m,sag_ratio,
                drape_ratio,mask_complete,static_hanging=True,diagnostics=False):
        d=np.asarray(down,float)
        if d.shape!=(2,) or not np.isfinite(d).all() or not np.isfinite(np.linalg.norm(d)) or np.linalg.norm(d)<1e-9:
            raise ValueError('finite nonzero down vector required')
        d=d/np.linalg.norm(d);axis=np.array([d[1],-d[0]])
        h,w=z.shape;yy,xx=np.indices(z.shape);u=xx*axis[0]+yy*axis[1];v=xx*d[0]+yy*d[1]
        silhouette=candidate_mask&~arm_mask&work_roi
        warnings=[]
        if not mask_complete:warnings.append('mask_completeness_unverified')
        if (silhouette&~valid).any():warnings.append('some_object_depth_unobserved')
        stages={'grasp':_stage('insufficient_input',value='unknown',reason='shape_does_not_verify_physical_grasp')}
        trace={};anchors=[];points=[];interiors=np.zeros_like(valid)
        if hand_rois is not None:
            for i,name in enumerate(('left','right')):
                box=np.asarray(hand_rois[name],float);x0,y0,x1,y1=box
                corners=np.array([[x0,y0],[x1-1,y0],[x0,y1-1],[x1-1,y1-1]])
                pu=float(np.mean(corners@axis));pv=float(np.max(corners@d));points.append((pu,pv))
                anchors.append((u>=np.min(corners@axis))&(u<=np.max(corners@axis))&(np.abs(v-pv)<=anchor_band_px))
                interiors|=(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)&~anchors[-1]
                if x0<0 or y0<0 or x1>w or y1>h:warnings.append(name+'_roi_truncated')
                if hand_visibility[i]!='visible':warnings.append(name+'_hand_'+hand_visibility[i])
        path_domain=silhouette&valid&~interiors
        # Candidate bands delimit the below-contact search, never the upper arm route.
        if len(points)==2 and abs(points[1][0]-points[0][0])>=8:
            (lu,lv),(ru,rv)=points;baseline=lv+(u-lu)*(rv-lv)/(ru-lu)
            path_domain &= v>=baseline-anchor_band_px
        parts,rejected,rejected_count=_graph_components(path_domain,z,depth_jump_m)
        if rejected_count:warnings.append('local_depth_edges_rejected')
        stages['depth_connectivity']=_stage('computed' if path_domain.any() else 'insufficient_input',
            usable_pixels=int(path_domain.sum()),component_sizes=[len(p) for p in parts],
            rejected_edge_count=rejected_count,depth_jump_m=float(depth_jump_m),inferred_pixels=0)
        candidate_records=[];spanning=[];attachment_records=[]
        for index,p in enumerate(parts):
            ys,xs=p.T;component=np.zeros_like(valid);component[ys,xs]=True
            hits=[int((component&a).sum()) for a in anchors]
            patches=[int(_patch_centers(component&a).sum()) for a in anchors]
            bands_complete=all(a.any() and np.all((valid&~arm_mask&work_roi)[a]) for a in anchors)
            supported=len(p)>=min_area and len(hits)==2 and all(n>=3 for n in hits) and (bands_complete or all(n>0 for n in patches))
            supported = supported and abs(points[1][0]-points[0][0])>=8 and not np.any(anchors[0]&anchors[1])
            if supported:spanning.append(component)
            candidate_records.append({'component':index,'pixels':len(p),'meets_min_area':len(p)>=min_area,
                'anchor_hits':hits,'local_patch_centers':patches,'spans_candidate_bands':supported})
        bridge='yes' if spanning else 'unknown'
        stages['observed_bridge']=_stage('computed' if anchors and path_domain.any() else 'insufficient_input',
            value=bridge,reference='caller_attachment_bands' if anchors else None,
            warnings=list(warnings),candidates=candidate_records,
            reason='observed_depth_continuous_path_between_visible_band_pixels' if spanning else 'no_supported_observed_pair',
            physical_contact_verified=False)
        path=np.logical_or.reduce(spanning) if spanning else np.zeros_like(valid)
        for i,a in enumerate(anchors):
            matching=[p for p in parts if len(p)>=min_area and int(a[p[:,0],p[:,1]].sum())>=3]
            if diagnostics:
                attachment_mask=np.zeros_like(valid)
                for p in matching:attachment_mask[p[:,0],p[:,1]]=True
                trace['feature_'+('left','right')[i]+'_drape_component_mask']=attachment_mask
            if matching:
                ys,xs=np.concatenate(matching).T;extent=float(max(0,np.quantile(v[ys,xs]-points[i][1],.95)))
                span=abs(points[1][0]-points[0][0]);ratio=extent/span if span>=8 else None
                attachment_records.append(_stage('computed',candidate=i,extent_px=extent,extent_to_span=ratio,
                    hangs=bool(ratio>=drape_ratio) if ratio is not None else None,hand_identity='caller_label_not_verified'))
            else:attachment_records.append(_stage('insufficient_input',candidate=i,extent_px=None))
        stages['attachment_drape']=_stage('computed' if any(x['status']=='computed' for x in attachment_records) else 'insufficient_input',
            candidates=attachment_records,warnings=list(warnings))
        ys,xs=np.nonzero(silhouette)
        if len(xs):
            width=float(np.ptp(u[ys,xs]));height=float(np.ptp(v[ys,xs]));top=float(v[ys,xs].min())
            edge=np.zeros_like(valid)
            if d[1]>0:edge[0]=True
            if d[1]<0:edge[-1]=True
            if d[0]>0:edge[:,0]=True
            if d[0]<0:edge[:,-1]=True
            upper_clipped=bool((edge&silhouette).any())
            stages['morphology']=_stage('computed',pixels=len(xs),width_px=width,down_extent_px=height,
                top_boundary_clipped=upper_clipped,extent_to_width=height/max(width,1),warnings=list(warnings))
            stages['hanging']=_stage('inferred' if static_hanging else 'computed',
                value='present' if height/max(width,1)>=drape_ratio else 'short_extent',
                extent_px=height,reference='visible_upper_contour',assumption='static_hanging' if static_hanging else None,
                warnings=list(warnings)+['visible_morphology_not_support_identity'])
            # Upper contour peaks offer hypotheses, never a unique physical support count.
            near=silhouette&(v<=top+max(2,.1*height));uu=u[near]
            upper_width=float(np.ptp(uu)) if len(uu) else 0.
            entry_bins=np.unique(np.floor(uu).astype(int))
            entry_lobes=int(1+np.count_nonzero(np.diff(entry_bins)>1)) if len(entry_bins) else 0
            preferred=2 if spanning or entry_lobes==2 else 1 if upper_width<.35*max(width,1) else None
            cu,cv=_envelope(silhouette,u,v)
            valley=float(np.quantile(cv,.9)-min(cv[0],cv[-1])) if len(cv) else 0.
            if preferred is None and not upper_clipped and width>=8 and valley>.08*width:preferred=2
            # Rays point away from the visible hanging body: upward or diagonal upward.
            ray_points=[]
            for target in ([float(np.median(uu))] if preferred==1 else [float(np.quantile(uu,.1)),float(np.quantile(uu,.9))]):
                local=near&(np.abs(u-target)<=max(2,.12*max(width,1)))
                if not local.any():continue
                pu=float(np.median(u[local]));pv=float(np.min(v[local]));body=silhouette&(v>=pv+max(2,.1*height))&(v<=pv+max(4,.3*height))
                body_u=float(np.median(u[body])) if body.any() else pu
                direction=(pu-body_u)*axis-max(4,.2*height)*d;direction/=np.linalg.norm(direction)
                ray_points.append({'origin_xy':(pu*axis+pv*d).tolist(),'direction_xy':direction.tolist(),
                    'location':'above_or_diagonal_above_visible_contour','distance_px':None})
            stages['support_hypotheses']=_stage('inferred' if static_hanging else 'insufficient_input',
                assumption='static_hanging' if static_hanging else None,preferred_count_hint=preferred if static_hanging else None,
                possible_counts=[1,2] if static_hanging else [],rays=ray_points if static_hanging else [],
                visible_attachment_candidates=2 if spanning else sum(x['status']=='computed' for x in attachment_records),
                unique_support_count=False,upper_entry_lobes=entry_lobes,hand_identity='unresolved',
                warnings=['hidden_supports_can_merge_or_share_one_visible_silhouette'])
        else:
            upper_clipped=False
            for name in ('morphology','hanging','support_hypotheses'):
                stages[name]=_stage('insufficient_input',reason='no_eligible_object_silhouette')
        # Sag is a measured contour/chord relation; uncertainty does not erase it.
        sag_mask=path if path.any() else silhouette
        su,sv=_envelope(sag_mask,u,v);sag=_stage('insufficient_input',value='unknown',reason='insufficient_visible_span')
        if len(su)>=2 and su[-1]-su[0]>=8:
            if spanning:
                lo,hi=sorted((points[0][0],points[1][0]));use=(su>=lo)&(su<=hi)
                ref='observed_candidate_band_chord';status='computed'
                line=points[0][1]+(su-points[0][0])*(points[1][1]-points[0][1])/(points[1][0]-points[0][0])
            else:
                lo,hi=su[0],su[-1];use=np.ones(len(su),bool);ref='visible_contour_endpoint_chord';status='inferred'
                line=sv[0]+(su-su[0])*(sv[-1]-sv[0])/(su[-1]-su[0])
            if use.any() and hi-lo>=8:
                drop=float(max(0,np.quantile(sv[use]-line[use],.9)));ratio=drop/(hi-lo)
                ident=bool(spanning) or not upper_clipped
                sag=_stage(status,value=('yes' if ratio>=sag_ratio else 'no') if ident else 'unknown',
                    reference=ref,drop_px=drop,span_px=float(hi-lo),ratio=float(ratio),
                    observed_span_fraction=min(1.,float(use.sum()/max(1,np.ceil(hi-lo)))),
                    candidate_band_chord_observed=bool(spanning),physical_support_chord_verified=False,threshold=float(sag_ratio),
                    warnings=list(warnings)+([] if spanning else ['support_chord_inferred_not_unique'])+
                    (['upper_contour_clipped_true_sag_unidentifiable'] if upper_clipped and not spanning else []))
                if diagnostics:trace['feature_sag_path_xy']=np.column_stack((su[use],sv[use]))@np.stack((axis,d))
        stages['sag']=sag
        if diagnostics:
            trace.update(feature_observed_bridge_mask=path,feature_depth_rejected_edge_mask=rejected,
                feature_eligible_silhouette_mask=silhouette,feature_usable_depth_mask=path_domain)
        return stages,trace
