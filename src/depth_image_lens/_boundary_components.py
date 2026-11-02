"""Bounded trail measurements on the existing compressed boundary graph.

Segments are edges. A junction region is a vertex with unmeasured internal travel;
these lengths must not be advertised as complete pixel-skeleton geodesics.
"""
import math
import numpy as np


def _adjacency(edges, vertices):
    adj = {v: [] for v in vertices}
    for i, (u, v, _) in enumerate(edges):
        adj[u].append((i, False, v))
        if u != v:
            adj[v].append((i, True, u))
    return adj


def _spanning(adj, root):
    parent = {root: None}
    order = [root]
    for u in order:
        for edge, reverse, v in adj[u]:
            if v not in parent:
                parent[v] = (u, edge, reverse)
                order.append(v)
    return parent, order


def _tree_path(parent, start, end):
    """Oriented spanning-tree path, also used to make a feasible lower bound."""
    ancestors = {start}
    u = start
    while parent[u] is not None:
        u = parent[u][0]
        ancestors.add(u)
    right = []
    v = end
    while v not in ancestors:
        previous, edge, reverse = parent[v]
        right.append((edge, reverse))
        v = previous
    left = []
    u = start
    while u != v:
        previous, edge, reverse = parent[u]
        left.append((edge, not reverse))
        u = previous
    return left + right[::-1]


def _euler(adj, root):
    used = set()
    cursor = dict.fromkeys(adj, 0)
    stack = [(root, None)]
    reverse_trail = []
    while stack:
        u, incoming = stack[-1]
        while cursor[u] < len(adj[u]) and adj[u][cursor[u]][0] in used:
            cursor[u] += 1
        if cursor[u] == len(adj[u]):
            stack.pop()
            if incoming is not None:
                reverse_trail.append(incoming)
        else:
            edge, reverse, v = adj[u][cursor[u]]
            cursor[u] += 1
            used.add(edge)
            stack.append((v, (edge, reverse)))
    return reverse_trail[::-1]


def _solve(edges, vertices, terminals, budget, max_search_edges):
    """Exact fast cases; otherwise bounded iterative edge-subset DFS.

    budget is shared across every component in one API call. Shortest-path
    diameter is never substituted for a cyclic graph's maximum trail.
    """
    closed = len(terminals) < 2
    mode = 'closed_trail' if closed else 'endpoint_trail'
    if not edges:
        return dict(mode=mode, status='no_edges', exact=True, reason='no_traversable_segments',
                    path=[], start=None, states=0)
    adj = _adjacency(edges, vertices)
    root = vertices[0]
    parent, order = _spanning(adj, root)
    if len(order) != len(vertices):
        raise ValueError('component is disconnected in the compressed graph')
    degrees = dict.fromkeys(vertices, 0)
    for u, v, _ in edges:
        degrees[u] += 1
        degrees[v] += 1
    odd = {v for v in vertices if degrees[v] % 2}
    if (closed and not odd) or (not closed and len(terminals) == 2 and odd == set(terminals)):
        start = root if closed else terminals[0]
        return dict(mode=mode, status='exact', exact=True, reason='euler_trail',
                    path=_euler(adj, start), start=start, states=0)
    tree = len(edges) == len(vertices) - 1
    if tree and closed:
        return dict(mode=mode, status='no_closed_trail', exact=True, reason='acyclic_without_endpoint_pair',
                    path=[], start=None, states=0)
    if tree:
        # Two tree traversals find the diameter restricted to terminal endpoints.
        def farthest(start):
            parents, visited = _spanning(adj, start)
            distances = {start: 0.}
            for v in visited[1:]:
                u, edge, _ = parents[v]
                distances[v] = distances[u] + edges[edge][2]
            end = max((v for v in terminals if v != start), key=lambda v: (distances[v], -v))
            return end, parents
        start, _ = farthest(terminals[0])
        end, parents = farthest(start)
        return dict(mode=mode, status='exact', exact=True, reason='tree_terminal_diameter',
                    path=_tree_path(parents, start, end), start=start, states=0)

    def length(path):
        return math.fsum(edges[i][2] for i, _ in path)

    # A valid representative is available even when exhaustive search is refused.
    if closed:
        tree_edges = {entry[1] for entry in parent.values() if entry is not None}
        i = next(i for i in range(len(edges)) if i not in tree_edges)
        u, v, _ = edges[i]
        best = [(i, False)] + _tree_path(parent, v, u)
        best_start = u
    else:
        best_start = terminals[0]
        parents, order = _spanning(adj, best_start)
        distances = {best_start: 0.}
        for v in order[1:]:
            u, edge, _ = parents[v]
            distances[v] = distances[u] + edges[edge][2]
        # This is explicitly a lower bound, not a shortest-path diameter result.
        end = max(terminals[1:], key=lambda v: (distances[v], -v))
        best = _tree_path(parents, best_start, end)
    best_length = length(best)
    if len(best) == len(edges):
        return dict(mode=mode, status='exact', exact=True, reason='all_segments_trail',
                    path=best, start=best_start, states=0)
    if len(edges) > max_search_edges:
        return dict(mode=mode, status='bounded', exact=False, reason='max_search_edges_exceeded',
                    path=best, start=best_start, states=0)

    starts = vertices if closed else terminals
    terminal_set = set(terminals)
    stack = [(start, start, 0, []) for start in reversed(starts)]
    seen = set()
    states = 0
    while stack:
        start, u, used, path = stack.pop()
        key = (start, u, used)
        if key in seen:
            continue
        if budget[0] == 0:
            return dict(mode=mode, status='bounded', exact=False, reason='max_search_states_exhausted',
                        path=best, start=best_start, states=states)
        budget[0] -= 1
        states += 1
        seen.add(key)
        admissible = bool(path) and (u == start if closed else u in terminal_set and u != start)
        if admissible:
            value = length(path)
            if value > best_length:
                best, best_start, best_length = path, start, value
            if len(path) == len(edges):
                return dict(mode=mode, status='exact', exact=True, reason='all_segments_trail',
                            path=path, start=start, states=states)
        for i, reverse, v in reversed(adj[u]):
            if not used & (1 << i):
                stack.append((start, v, used | (1 << i), path + [(i, reverse)]))
    return dict(mode=mode, status='exact', exact=True, reason='exhaustive_edge_trails',
                path=best, start=best_start, states=states)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    return int(value)


def _points(value, shape):
    xy = np.asarray(value)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0 or xy.dtype.kind not in 'iu':
        raise ValueError('graph points must be nonempty integer Nx2 coordinates')
    if (xy < 0).any() or (xy[:, 0] >= shape[1]).any() or (xy[:, 1] >= shape[0]).any():
        raise ValueError('graph point outside result image')
    return xy


def measure_graph(graph, shape, kind, max_search_states, max_search_edges):
    """Return detached JSON-compatible component measurements without changing a graph."""
    for name, value, limit in [('max_search_states', max_search_states, 1_000_000),
                                ('max_search_edges', max_search_edges, 64)]:
        if type(value) is not int or not 0 <= value <= limit:
            raise ValueError(f'{name} must be a Python integer in 0..{limit}')
    result = dict(kind=kind, status='computed', components=[],
                  parameters=dict(max_search_states=max_search_states, max_search_edges=max_search_edges),
                  scope='compressed_segment_graph_not_physical_boundary_or_complete_pixel_geodesic',
                  warnings=['junction_region_internal_travel_is_unmeasured_and_not_added',
                            'terminals_can_be_caused_by_threshold_roi_crop_or_missing_observations',
                            'inferred_gap_segments_are_included_when_present',
                            'pixel_lengths_depend_on_image_resolution'])
    if not graph or graph.get('status') != 'computed':
        result.update(status='insufficient', reason='trace_boundaries_required' if not graph else graph.get('reason', 'topology_insufficient'))
        return result
    groups = {}
    node_lookup = {}
    segment_ids = set()
    for node in graph['nodes']:
        nid = _integer(node['id'], 'node id')
        cid = _integer(node['component'], 'component id')
        if nid in node_lookup:
            raise ValueError('duplicate node id')
        if node['kind'] not in ('endpoint', 'corner', 'junction', 'isolated'):
            raise ValueError('unknown node kind')
        _points(node['pixels_xy'], shape)
        record = dict(node, id=nid, component=cid)
        node_lookup[nid] = record
        groups.setdefault(cid, dict(nodes=[], segments=[]))['nodes'].append(record)
    for segment in graph['segments']:
        sid = _integer(segment['id'], 'segment id')
        cid = _integer(segment['component'], 'component id')
        if sid in segment_ids:
            raise ValueError('duplicate segment id')
        segment_ids.add(sid)
        xy = _points(segment['points_xy'], shape)
        length = segment['length_px']
        if isinstance(length, (bool, np.bool_)) or not np.isscalar(length) or not np.isfinite(length) or length < 0:
            raise ValueError('finite nonnegative segment length required')
        if len(segment['evidence']) != len(xy) or any(e not in ('strong', 'weak', 'inferred') for e in segment['evidence']):
            raise ValueError('segment evidence must match its points')
        u, v = segment['start_node'], segment['end_node']
        if u is None or v is None:
            if not (u is None and v is None and segment['closed'] and np.array_equal(xy[0], xy[-1])):
                raise ValueError('null graph endpoints require a closed standalone loop')
        else:
            for nid in (u, v):
                nid = _integer(nid, 'segment endpoint')
                if nid not in node_lookup or node_lookup[nid]['component'] != cid:
                    raise ValueError('segment endpoint missing or belongs to another component')
        record = dict(segment, id=sid, component=cid, length_px=float(length),
                      start_node=None if u is None else int(u), end_node=None if v is None else int(v))
        groups.setdefault(cid, dict(nodes=[], segments=[]))['segments'].append(record)
    budget = [max_search_states]
    for cid, group in sorted(groups.items()):
        nodes = sorted(group['nodes'], key=lambda n: n['id'])
        segments = sorted(group['segments'], key=lambda s: s['id'])
        coords = np.concatenate([_points(n['pixels_xy'], shape) for n in nodes]
                                + [_points(s['points_xy'], shape) for s in segments])
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        indices = {n['id']: i for i, n in enumerate(nodes)}
        vertices = list(range(len(nodes)))
        edges = []
        for segment in segments:
            if segment['start_node'] is None:
                u = v = len(vertices)
                vertices.append(u)  # Internal only; do not invent a public endpoint.
            else:
                u, v = indices[segment['start_node']], indices[segment['end_node']]
            edges.append((u, v, float(segment['length_px'])))
        terminals = [indices[n['id']] for n in nodes if n['kind'] == 'endpoint']
        degrees = dict.fromkeys(vertices, 0)
        for u, v, _ in edges:
            degrees[u] += 1
            degrees[v] += 1
        if any(degrees[v] != 1 for v in terminals):
            raise ValueError('terminal endpoint must have degree one')
        solved = _solve(edges, vertices, terminals, budget, max_search_edges)
        parts = []
        for i, reverse in solved['path']:
            segment = segments[i]
            points = segment['points_xy'][::-1] if reverse else segment['points_xy']
            evidence = segment['evidence'][::-1] if reverse else segment['evidence']
            parts.append(dict(segment_id=segment['id'], reversed=reverse,
                              start_node=segment['end_node'] if reverse else segment['start_node'],
                              end_node=segment['start_node'] if reverse else segment['end_node'],
                              points_xy=[list(map(int, p)) for p in points], evidence=list(evidence),
                              length_px=float(segment['length_px'])))
        joins = []
        pairs = list(zip(range(len(parts) - 1), range(1, len(parts))))
        if parts and solved['mode'] == 'closed_trail':
            pairs.append((len(parts) - 1, 0))
        for a, b in pairs:
            end, start = parts[a]['points_xy'][-1], parts[b]['points_xy'][0]
            if end != start:
                joins.append(dict(from_part=a, to_part=b, node_id=parts[a]['end_node'],
                                  from_xy=end.copy(), to_xy=start.copy(), length_included_px=0.,
                                  status='junction_travel_unmeasured'))
        feasible = bool(parts) or solved['status'] == 'no_edges'
        lower = math.fsum(p['length_px'] for p in parts) if feasible else None
        maximum = lower if solved['exact'] else None
        total = math.fsum(e[2] for e in edges)
        component = dict(id=cid, bbox_xyxy=[int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])],
                         span_xy_px=(hi - lo).tolist(), bbox_size_px=(hi - lo + 1).tolist(),
                         node_ids=[n['id'] for n in nodes], segment_ids=[s['id'] for s in segments],
                         endpoint_node_ids=[n['id'] for n in nodes if n['kind'] == 'endpoint'],
                         junction_node_ids=[n['id'] for n in nodes if n['kind'] == 'junction'],
                         touches_image_boundary=bool((lo == 0).any() or hi[0] == shape[1]-1 or hi[1] == shape[0]-1),
                         total_segment_length_px=total, path_kind=solved['mode'],
                         endpoint_pair_available=len(terminals) >= 2, status=solved['status'], exact=solved['exact'],
                         reason=solved['reason'], search_states=solved['states'],
                         max_trail_length_px=maximum, lower_bound_length_px=lower,
                         upper_bound_length_px=maximum if solved['exact'] else total,
                         max_endpoint_trail_length_px=maximum if solved['mode'] == 'endpoint_trail' else None,
                         max_closed_trail_length_px=maximum if solved['mode'] == 'closed_trail' and parts else None,
                         contains_inferred=any('inferred' in s['evidence'] for s in segments),
                         contains_weak=any('weak' in s['evidence'] for s in segments),
                         trail=dict(parts=parts, junction_transitions=joins,
                                    start_node=parts[0]['start_node'] if parts else None,
                                    end_node=parts[-1]['end_node'] if parts else None,
                                    start_xy=parts[0]['points_xy'][0].copy() if parts else None,
                                    end_xy=parts[-1]['points_xy'][-1].copy() if parts else None,
                                    closed=solved['mode'] == 'closed_trail' and bool(parts),
                                    uses_inferred=any('inferred' in p['evidence'] for p in parts),
                                    uses_weak=any('weak' in p['evidence'] for p in parts)))
        result['components'].append(component)
    result['summary'] = dict(component_count=len(groups), search_states=max_search_states-budget[0],
                             bounded_components=sum(not c['exact'] for c in result['components']))
    result['domain'] = graph.get('domain')
    return result
