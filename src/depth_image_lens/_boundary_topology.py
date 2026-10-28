"""Deterministic, optional topology of image evidence; no physical identity inference."""
from collections import Counter
import math
import numpy as np
from .debug import _opencv


def _thin(mask):
    """Two-subiteration Zhang-Suen binary thinning, implemented with NumPy."""
    a=np.pad(mask.astype(np.uint8),1)
    for _ in range(max(mask.shape)):
        changed=False
        for phase in (0,1):
            p=[a[:-2,1:-1],a[:-2,2:],a[1:-1,2:],a[2:,2:],
               a[2:,1:-1],a[2:,:-2],a[1:-1,:-2],a[:-2,:-2]]
            count=sum(p);trans=sum((p[i]==0)&(p[(i+1)%8]!=0) for i in range(8))
            constraints=(p[0]*p[2]*p[4]==0)&(p[2]*p[4]*p[6]==0) if phase==0 else (p[0]*p[2]*p[6]==0)&(p[0]*p[4]*p[6]==0)
            remove=(a[1:-1,1:-1]!=0)&(count>=2)&(count<=6)&(trans==1)&constraints
            if remove.any():a[1:-1,1:-1][remove]=0;changed=True
        if not changed:break
    return a[1:-1,1:-1].astype(bool)


def _adjacency(mask):
    pixels={(int(y),int(x)) for y,x in np.argwhere(mask)}
    graph={}
    for y,x in sorted(pixels):
        neighbors=[]
        for dy,dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            q=(y+dy,x+dx)
            if q not in pixels:continue
            # Suppress diagonal shortcuts where an orthogonal route already exists.
            if dy and dx and ((y,x+dx) in pixels or (y+dy,x) in pixels):continue
            neighbors.append(q)
        graph[(y,x)]=sorted(neighbors)
    return graph


def _length(points):
    return float(np.linalg.norm(np.diff(np.asarray(points,float),axis=0),axis=1).sum()) if len(points)>1 else 0.


def _angle(a,b):
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/denom,-1,1)))) if denom>1e-12 else None


def _outward(graph,end,window):
    path=[end];previous=None;current=end
    while _length(path)<window:
        options=[q for q in graph[current] if q!=previous]
        if len(options)!=1:break
        q=options[0];path.append(q);previous,current=current,q
        if len(graph[q])!=2:break
    if _length(path)<2:return None
    return np.asarray(end,float)-np.asarray(path[-1],float)


def _reconnect(skeleton,domain,max_gap,angle_limit,window):
    cv=_opencv();inferred=np.zeros_like(skeleton);connections=[];reasons=Counter()
    if not max_gap:return inferred,connections,dict(reasons)
    graph=_adjacency(skeleton);ends=[p for p in graph if len(graph[p])==1]
    tangents={p:_outward(graph,p,window) for p in ends}
    _,regions=cv.connectedComponents(domain.astype(np.uint8),connectivity=8)
    radius=max_gap+1;bins={}
    for p in ends:bins.setdefault((p[0]//radius,p[1]//radius),[]).append(p)
    options={p:[] for p in ends};paths={};angles={}
    occupied=set(graph)
    for p in ends:
        by,bx=p[0]//radius,p[1]//radius
        candidates=[q for dy in (-1,0,1) for dx in (-1,0,1) for q in bins.get((by+dy,bx+dx),[]) if q>p]
        for q in candidates:
            delta=np.asarray(q,float)-p;distance=float(np.linalg.norm(delta))
            if distance<=1 or distance>radius:continue
            if tangents[p] is None or tangents[q] is None:reasons['insufficient_tangent_support']+=1;continue
            aa=_angle(tangents[p],delta);bb=_angle(tangents[q],-delta)
            if max(aa,bb)>angle_limit:reasons['direction_mismatch']+=1;continue
            count=max(abs(q[0]-p[0]),abs(q[1]-p[1]))+1
            line=[(int(y),int(x)) for y,x in np.rint(np.linspace(p,q,count)).astype(int)]
            if regions[p]==0 or regions[p]!=regions[q] or not all(domain[v] for v in line):reasons['unobserved_or_other_domain']+=1;continue
            internal=set(line[1:-1]);missing=internal-occupied
            if not missing or len(missing)>max_gap:continue
            # No crossing or side contact with a different skeleton section.
            allowed={(v[0]+dy,v[1]+dx) for v in (p,q) for dy in (-1,0,1) for dx in (-1,0,1)}
            neighbors={(y+dy,x+dx) for y,x in internal for dy in (-1,0,1) for dx in (-1,0,1)}
            if (neighbors&occupied)-allowed:reasons['crossing_or_side_contact']+=1;continue
            options[p].append(q);options[q].append(p);paths[(p,q)]=line;angles[(p,q)]=[aa,bb]
    pairs=[(p,qs[0]) for p,qs in options.items() if len(qs)==1 and len(options[qs[0]])==1 and p<qs[0]]
    reasons['ambiguous_endpoints']=sum(len(qs)>1 for qs in options.values())
    # Reject both proposed connections if they would cross or touch each other.
    blocked=set()
    for i,pair in enumerate(pairs):
        neighborhood={(y+dy,x+dx) for y,x in paths[pair][1:-1] for dy in (-1,0,1) for dx in (-1,0,1)}
        for other in pairs[i+1:]:
            if neighborhood.intersection(paths[other][1:-1]):blocked.update((pair,other))
    reasons['conflicting_connections']=len(blocked)
    for pair in pairs:
        if pair in blocked:continue
        line=paths[pair];pixels=[v for v in line if not skeleton[v]]
        for v in pixels:inferred[v]=True
        connections.append(dict(id=len(connections)+1,endpoints_xy=[list(reversed(v)) for v in pair],
                                pixels_xy=[list(reversed(v)) for v in pixels],missing_pixels=len(pixels),
                                endpoint_distance_px=_length([pair[0],pair[1]]),direction_errors_deg=angles[pair],
                                status='inferred_connection_not_observed_boundary'))
    return inferred,connections,dict(reasons)


def _paths(skeleton):
    graph=_adjacency(skeleton);special={p for p,qs in graph.items() if len(qs)!=2}
    junction={p for p in special if len(graph[p])>2};groups=[]
    while junction:
        seed=min(junction);junction.remove(seed);group={seed};stack=[seed]
        while stack:
            for q in graph[stack.pop()]:
                if q in junction:junction.remove(q);group.add(q);stack.append(q)
        groups.append(('junction',sorted(group)))
    groups += [('endpoint' if len(graph[p])==1 else 'isolated',[p]) for p in sorted(special) if len(graph[p])<=1]
    groups.sort(key=lambda pair:pair[1][0]);nodes=[];membership={}
    for kind,points in groups:
        node_id=len(nodes)+1;center=np.mean(points,axis=0);rep=min(points,key=lambda p:(np.linalg.norm(np.asarray(p)-center),p))
        nodes.append(dict(id=node_id,kind=kind,xy=list(reversed(rep)),pixels_xy=[list(reversed(p)) for p in points]))
        membership.update({p:node_id for p in points})
    visited=set();paths=[]
    edge_key=lambda a,b:tuple(sorted((a,b)))
    for p in sorted(membership):
        for q in graph[p]:
            if membership.get(q)==membership[p] or edge_key(p,q) in visited:continue
            path=[p,q];visited.add(edge_key(p,q));previous,current=p,q
            while current not in membership:
                nxt=[v for v in graph[current] if v!=previous]
                if len(nxt)!=1:break
                v=nxt[0];key=edge_key(current,v)
                if key in visited:break
                path.append(v);visited.add(key);previous,current=current,v
            paths.append((path,membership[p],membership.get(current),False))
    # Remaining degree-two components are closed loops, with no invented endpoint.
    for p in sorted(graph):
        for q in graph[p]:
            key=edge_key(p,q)
            if key in visited or (p in membership and q in membership):continue
            path=[p,q];visited.add(key);previous,current=p,q
            while current!=p:
                nxt=[v for v in graph[current] if v!=previous]
                if len(nxt)!=1:break
                v=nxt[0];key=edge_key(current,v)
                if key in visited:break
                path.append(v);visited.add(key);previous,current=current,v
            paths.append((path,None,None,current==p))
    return nodes,paths


def _corner_indices(path,closed,window,threshold):
    points=np.asarray(path,float);points=points[:-1] if closed else points
    n=len(points)
    if n<3:return []
    if closed:
        extended=np.concatenate([points,points,points]);offset=n
    else:extended=points;offset=0
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(extended,axis=0),axis=1))]
    candidates=[]
    for i in range(n):
        j=i+offset
        if arc[j]<window or arc[-1]-arc[j]<window:continue
        left=max(0,int(np.searchsorted(arc,arc[j]-window,side='right')-1));right=min(len(arc)-1,int(np.searchsorted(arc,arc[j]+window)))
        turn=_angle(extended[j]-extended[left],extended[right]-extended[j])
        if turn is not None and turn>=threshold:candidates.append((i,turn,float(arc[j])))
    selected=[];total=_length(path)
    for item in sorted(candidates,key=lambda c:(-c[1],c[0])):
        if any(min(abs(item[2]-q[2]),total-abs(item[2]-q[2]))<=window if closed else abs(item[2]-q[2])<=window for q in selected):continue
        selected.append(item)
    return sorted([(i,a) for i,a,_ in selected])


def _junction_groups(nodes,segments,parameters):
    """Optional geometric grouping hypotheses; never modify the observed graph."""
    limit=parameters['junction_cluster_px'];errors=Counter();groups=[]
    junctions={n['id']:n for n in nodes if n['kind']=='junction'}
    result=dict(status='disabled' if not limit else 'computed',groups=groups,
                grouped_junction_count=len(junctions),rejections={})
    if not limit:return result
    by_id={s['id']:s for s in segments};candidates=[]
    tolerance=parameters['junction_cluster_residual_px'];angle_limit=parameters['junction_cluster_angle_deg']
    window=parameters['tangent_window_px']
    for link in segments:
        u,v=link['start_node'],link['end_node']
        if u==v or u not in junctions or v not in junctions or link['length_px']>limit:continue
        if 'inferred' in link['evidence']:errors['interpolated_connector']+=1;continue
        pair=[junctions[u],junctions[v]]
        if any(len(n['arms'])!=3 for n in pair):errors['not_two_three_arm_junctions']+=1;continue
        arms=[a for n in pair for a in n['arms'] if a['segment_id']!=link['id']]
        if len(arms)!=4 or len({a['segment_id'] for a in arms})!=4:errors['shared_or_looping_external_arm']+=1;continue
        fits=[]
        for arm in arms:
            segment=by_id[arm['segment_id']]
            points=np.asarray(segment['points_xy'],float);evidence=segment['evidence']
            if arm['end']=='end':points=points[::-1];evidence=evidence[::-1]
            arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
            skip=max(limit,window);selected=(arc>=skip)&(arc<=skip+2*window)
            sample=points[selected]
            if len(sample)<4 or _length(sample)<window or any(e=='inferred' for e,k in zip(evidence,selected) if k):break
            center=sample.mean(axis=0);_,_,basis=np.linalg.svd(sample-center,full_matrices=False)
            direction=basis[0]
            if np.dot(direction,sample[-1]-sample[0])<0:direction=-direction
            normal=np.array([-direction[1],direction[0]])
            residual=float(np.max(np.abs((sample-center)@normal)))
            if residual>tolerance:break
            fits.append((center,direction,normal,residual))
        if len(fits)!=4:errors['insufficient_or_curved_observed_arm_support']+=1;continue
        pairings=[]
        for pairs in (((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))):
            angles=[180-_angle(fits[a][1],fits[b][1]) for a,b in pairs]
            if max(angles)<=angle_limit:pairings.append((pairs,angles))
        if len(pairings)!=1:errors['opposing_directions_ambiguous_or_inconsistent']+=1;continue
        normals=np.array([f[2] for f in fits]);offsets=np.array([np.dot(f[0],f[2]) for f in fits])
        center,_,rank,_=np.linalg.lstsq(normals,offsets,rcond=None)
        if rank<2 or not np.isfinite(center).all():errors['degenerate_intersection']+=1;continue
        residuals=np.abs(normals@center-offsets);connector=np.asarray(link['points_xy'])
        inside=np.all(center>=connector.min(axis=0)-tolerance) and np.all(center<=connector.max(axis=0)+tolerance)
        if not inside or max(residuals)>tolerance or any(np.dot(f[0]-center,f[1])<0 for f in fits):
            errors['no_common_local_intersection']+=1;continue
        candidates.append(dict(nodes=[u,v],internal_segments=[link['id']],external_arms=arms,
            center_xy=center.tolist(),connector_length_px=link['length_px'],
            line_residuals_px=[f[3] for f in fits],intersection_residuals_px=residuals.tolist(),
            opposite_arm_pairs=[list(p) for p in pairings[0][0]],direction_errors_deg=pairings[0][1],
            status='inferred_junction_group_not_physical_identity'))
    usage=Counter(n for c in candidates for n in c['nodes'])
    for candidate in candidates:
        if any(usage[n]>1 for n in candidate['nodes']):errors['ambiguous_neighbor_group']+=1;continue
        groups.append(dict(id=len(groups)+1,**candidate))
    result.update(grouped_junction_count=len(junctions)-len(groups),rejections=dict(errors))
    return result


def trace_evidence(strong,weak,domain,parameters,measurements=None):
    """Return local graph IDs and masks without replacing measured evidence."""
    cv=_opencv();weak=(weak|strong)&domain
    count,labels=cv.connectedComponents(weak.astype(np.uint8),connectivity=8)
    selected=np.zeros(count,bool);selected[np.unique(labels[strong&domain])]=True;selected[0]=False
    retained=selected[labels]
    if retained.sum()>250_000:
        empty=np.zeros_like(retained)
        return dict(status='insufficient',reason='topology_retained_pixel_budget_exceeded_resize_or_restrict_domain',nodes=[],segments=[],connections=[],counts=dict(retained_pixels=int(retained.sum())),warnings=['original_measured_outputs_are_preserved']),dict(retained_mask=retained,observed_centerline_mask=empty,inferred_mask=empty.copy())
    observed=_thin(retained)
    if observed.sum()>50_000:
        return dict(status='insufficient',reason='topology_centerline_pixel_budget_exceeded_resize_or_restrict_domain',nodes=[],segments=[],connections=[],counts=dict(observed_centerline_pixels=int(observed.sum())),warnings=['original_measured_outputs_are_preserved']),dict(retained_mask=retained,observed_centerline_mask=observed,inferred_mask=np.zeros_like(observed))
    inferred,connections,rejections=_reconnect(observed,domain,parameters['max_gap_px'],parameters['gap_angle_deg'],parameters['tangent_window_px'])
    skeleton=observed|inferred;nodes,paths=_paths(skeleton);segments=[]
    _,component_ids=cv.connectedComponents(skeleton.astype(np.uint8),connectivity=8)
    for node in nodes:
        x,y=node['xy'];node['component']=int(component_ids[y,x])
    for path,start,end,closed in paths:
        corners=_corner_indices(path,closed,parameters['corner_window_px'],parameters['corner_turn_deg'])
        corner_nodes={}
        for idx,angle in corners:
            y,x=path[idx];node_id=len(nodes)+1;corner_nodes[idx]=node_id
            nodes.append(dict(id=node_id,kind='corner',xy=[x,y],pixels_xy=[[x,y]],turn_deg=angle,component=int(component_ids[y,x]),inferred=bool(inferred[y,x])))
        if closed and corners:
            base=path[:-1];cuts=[i for i,_ in corners]
            pieces=[(base[a:b+1] if b>a else base[a:]+base[:b+1],corner_nodes[a],corner_nodes[b],False) for a,b in zip(cuts,cuts[1:]+cuts[:1])]
        elif closed:pieces=[(path,None,None,True)]
        else:
            cuts=[0]+[i for i,_ in corners]+[len(path)-1]
            ids=[start]+[corner_nodes[i] for i,_ in corners]+[end]
            pieces=[(path[a:b+1],u,v,False) for a,b,u,v in zip(cuts,cuts[1:],ids,ids[1:])]
        for points,u,v,loop in pieces:
            length=_length(points);ys,xs=np.asarray(points,int).T
            evidence=['inferred' if inferred[y,x] else 'strong' if strong[y,x] else 'weak' for y,x in points]
            values={}
            for key,array in (measurements or {}).items():
                a=array[ys,xs];ok=np.isfinite(a)&~inferred[ys,xs]
                values[key]=dict(mean=float(a[ok].mean()),p95=float(np.quantile(a[ok],.95))) if ok.any() else None
            segments.append(dict(id=len(segments)+1,component=int(component_ids[points[0]]),start_node=u,end_node=v,closed=loop,points_xy=[[x,y] for y,x in points],evidence=evidence,length_px=length,
                                 status='candidate' if length>=parameters['min_path_length_px'] else 'short_path',measurements=values))
    for node in nodes:
        attached=[s for s in segments if node['id'] in (s['start_node'],s['end_node'])];node['segment_ids']=[s['id'] for s in attached]
        if node['kind']=='junction':
            # A loop returning to one junction contributes two distinct arms.
            arms=[];vectors=[]
            for segment in attached:
                for end in ('start_node','end_node'):
                    if segment[end]!=node['id']:continue
                    points=segment['points_xy'] if end=='start_node' else list(reversed(segment['points_xy']))
                    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(np.asarray(points,float),axis=0),axis=1))]
                    k=min(int(np.searchsorted(arc,parameters['tangent_window_px'])),len(points)-1)
                    vectors.append(np.asarray(points[k],float)-points[0])
                    arms.append(dict(segment_id=segment['id'],end=end.removesuffix('_node')))
            node['arms']=arms
            node['branch_angles_deg']=[dict(segments=[arms[i]['segment_id'],arms[j]['segment_id']],
                arms=[i,j],angle_deg=_angle(a,b)) for i,a in enumerate(vectors) for j,b in enumerate(vectors) if i<j]
    data=dict(status='computed',nodes=nodes,segments=segments,connections=connections,
              counts=dict(retained_pixels=int(retained.sum()),observed_centerline_pixels=int(observed.sum()),inferred_pixels=int(inferred.sum()),components=int(component_ids.max()),segments=len(segments),corners=sum(n['kind']=='corner' for n in nodes),junctions=sum(n['kind']=='junction' for n in nodes)),gap_rejections=rejections,
              warnings=['graph_ids_are_local_not_physical_or_temporal_identity','weak_tracking_can_join_texture_or_shadow','junction_branch_correspondence_is_unassigned','pixel_skeleton_is_resolution_dependent'])
    data['junction_grouping']=_junction_groups(nodes,segments,parameters)
    return data,dict(retained_mask=retained,observed_centerline_mask=observed,inferred_mask=inferred)
