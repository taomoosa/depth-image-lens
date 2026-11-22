"""Bounded missing-depth links between two independently coherent measured flanks."""
from dataclasses import dataclass

import numpy as np


def validate_gap_options(max_px, flank_depth_jump_m, min_lines, cross_depth_jump_m):
    if type(max_px) is not int or not 0 <= max_px <= 8:
        raise ValueError('invalid_gap_max_px must be an integer in 0..8')
    if type(min_lines) is not int or not 2 <= min_lines <= 128:
        raise ValueError('invalid_gap_min_lines must be an integer in 2..128')

    def checked(value, name):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value <= 0):
            raise ValueError(name + ' must be finite and positive')
        return float(value)

    return dict(invalid_gap_max_px=max_px, invalid_gap_min_lines=min_lines,
                invalid_gap_flank_depth_jump_m=checked(flank_depth_jump_m, 'invalid_gap_flank_depth_jump_m'),
                invalid_gap_depth_jump_m=(None if cross_depth_jump_m is None else
                                          checked(cross_depth_jump_m, 'invalid_gap_depth_jump_m')))


def _depth_edge(a, b, limit):
    tol = 4*np.finfo(float).eps*np.maximum(np.maximum(np.abs(a), np.abs(b)), limit)
    with np.errstate(invalid='ignore', over='ignore'):
        return np.abs(a-b) <= limit+tol


@dataclass(frozen=True)
class _Proposal:
    ids: tuple
    axis: str
    runs: tuple  # orthogonal line, first missing coordinate, exclusive end

    def inferred_mask(self, shape):
        inferred = np.zeros(shape, bool)
        add = inferred if self.axis == 'x' else inferred.T
        for row, lo, hi in self.runs:
            add[row, lo:hi] = True
        return inferred

    def crossing_summary(self, depth):
        z = depth if self.axis == 'x' else depth.T
        differences = [abs(float(z[row, lo-1])-float(z[row, hi])) for row, lo, hi in self.runs]
        return dict(supporting_lines=len(self.runs),
                    median_cross_gap_depth_step_m=float(np.median(differences)),
                    max_missing_run_px=max(hi-lo for _,lo,hi in self.runs))


def _aligned_flanks(z, prev, next_run, limit):
    # Track the same flank on adjacent scanlines. Compare only nearby measured
    # locations, never the left and right endpoints of a missing span.
    py, pl, ph = prev
    ny, nl, nh = next_run
    if abs(pl-nl) > 3 or abs(ph-nh) > 3:
        return False
    left = min(pl, nl)-1
    right = max(ph, nh)
    # If the user zigzags by 3px, compare neighboring pixels at the support
    # edge rather than sampling beyond each guaranteed 3-pixel flank.
    lx0, lx1 = max(left,pl-3), max(left,nl-3)
    rx0, rx1 = min(right,ph+2), min(right,nh+2)
    return (_depth_edge(z[py,lx0], z[ny,lx1], limit)
            and _depth_edge(z[py,rx0], z[ny,rx1], limit))


def gap_proposals(depth, labels, missing, support, *, max_px, flank_depth_jump_m,
                  min_lines, cross_depth_jump_m=None):
    """Find same-pair tracks of nearby parallel runs through real missing Z.

    Each flank must contain three measured pixels of its own original
    component. Depth must be locally coherent within each flank, but a step
    between flanks is allowed unless the caller sets the old optional cap.
    Optional silhouette support restricts candidates; it never supplies Z.
    """
    if not np.any(missing if support is None else missing & support):
        return
    for axis, z, ids, valid_gap, fg in (
            ('x', depth, labels, missing, support),
            ('y', depth.T, labels.T, missing.T, None if support is None else support.T)):
        if z.shape[1] < 7:
            continue
        allowed = valid_gap.copy() if fg is None else valid_gap & fg
        # No fill at frame edges; the last/first lines may be clipped.
        allowed[[0, -1], :] = False
        measured = ids > 0
        if fg is not None:
            measured &= fg  # The immediate measured endpoints lie in the allowed silhouette.
        buckets = {}
        for span in range(1, min(max_px, z.shape[1]-6) + 1):
            n = z.shape[1] - span - 1
            left, right = ids[:, :n], ids[:, span+1:]
            flanks = measured[:, :n] & measured[:, span+1:] & (left != right)
            for offset in range(1, span+1):
                flanks &= allowed[:, offset:offset+n]
            rows, starts = np.nonzero(flanks)
            bounded = (starts >= 2) & (starts+span+3 < z.shape[1])
            rows, starts = rows[bounded], starts[bounded]
            if not len(rows):
                continue
            ends = starts+span+1
            a, b = ids[rows,starts], ids[rows,ends]
            compatible = np.ones(len(rows), bool)
            for k in (1, 2):
                compatible &= (ids[rows,starts-k] == a) & (ids[rows,ends+k] == b)
                compatible &= _depth_edge(z[rows,starts-k], z[rows,starts-k+1], flank_depth_jump_m)
                compatible &= _depth_edge(z[rows,ends+k], z[rows,ends+k-1], flank_depth_jump_m)
            if cross_depth_jump_m is not None:
                compatible &= _depth_edge(z[rows,starts], z[rows,ends], cross_depth_jump_m)
            for row, start, aa, bb in zip(rows[compatible], starts[compatible], a[compatible], b[compatible]):
                key = (int(aa),int(bb))
                buckets.setdefault(key, []).append((int(row), int(start)+1, int(start)+span+1))
        for pair, runs in sorted(buckets.items()):
            # Gaps must touch on adjacent lines; two independent breaks
            # cannot be concatenated merely because they share component IDs.
            tracks = []
            active, current, used = {}, {}, set()
            last_row = -2
            for row, lo, hi in sorted(runs):
                if row != last_row:
                    active, current = (current if row == last_row+1 else {}), {}
                    used = set()
                    last_row = row
                item = (row, lo, hi)
                # Inclusive endpoint indices let adjacent slanted gaps touch.
                neighbors = {i for x in range(lo, hi+1) for i in active.get(x, ())}
                choices = [i for i in sorted(neighbors-used)
                           if _aligned_flanks(z, tracks[i][-1], item, flank_depth_jump_m)]
                if choices:
                    idx = choices[0]
                    tracks[idx].append(item)
                else:
                    idx = len(tracks)
                    tracks.append([item])
                used.add(idx)
                for x in range(lo, hi+1):
                    current.setdefault(x, []).append(idx)
            for track in tracks:
                if len(track) >= min_lines:
                    yield _Proposal(pair, axis, tuple(track))


def result_detail(status, **data):
    return dict(status=status, candidate=False, inferred_gap_pixels=0,
                scope='locally_coherent_flank_missing_depth_hypothesis', **data)
