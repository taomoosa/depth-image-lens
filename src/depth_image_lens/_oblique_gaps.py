"""Bounded, independently supported missing-depth hypotheses, never observations."""
from dataclasses import dataclass

import numpy as np


def validate_gap_options(max_px, depth_jump_m, min_lines):
    if type(max_px) is not int or not 0 <= max_px <= 8:
        raise ValueError('invalid_gap_max_px must be an integer in 0..8')
    if type(min_lines) is not int or not 2 <= min_lines <= 128:
        raise ValueError('invalid_gap_min_lines must be an integer in 2..128')
    if (isinstance(depth_jump_m, (bool, np.bool_)) or not np.isscalar(depth_jump_m)
            or not isinstance(depth_jump_m, (int, float, np.integer, np.floating))
            or not np.isfinite(depth_jump_m) or depth_jump_m <= 0):
        raise ValueError('invalid_gap_depth_jump_m must be finite and positive')
    return dict(invalid_gap_max_px=max_px, invalid_gap_min_lines=min_lines,
                invalid_gap_depth_jump_m=float(depth_jump_m))


@dataclass(frozen=True)
class _Proposal:
    ids: tuple
    axis: str
    runs: tuple  # orthogonal line, first missing coordinate, exclusive end

    def build(self, depth):
        """Local scratch depths for graph continuity only, never returned."""
        inferred = np.zeros(depth.shape, bool)
        scratch = depth.copy()
        add, values = (inferred, scratch) if self.axis == 'x' else (inferred.T, scratch.T)
        for row, lo, hi in self.runs:
            add[row, lo:hi] = True
            values[row, lo:hi] = np.linspace(values[row, lo-1], values[row, hi], hi-lo+2)[1:-1]
        return inferred, scratch


def gap_proposals(depth, labels, missing, support, *, max_px, depth_jump_m, min_lines):
    """Find same-pair, adjacent parallel gap runs between measured fragments.

    Only real missing depths in the caller's allowed domain can be traversed.
    Both flanks must also be covered by the independent object support; each
    proposal links exactly two measured labels, with no multi-fragment chains.
    Diagonal cuts can shift across adjacent lines; only horizontal/vertical
    traversals are used.
    """
    if not np.any(missing & support):
        return
    for axis, z, ids, valid_gap, fg in (
            ('x', depth, labels, missing, support),
            ('y', depth.T, labels.T, missing.T, support.T)):
        if z.shape[1] < 3:
            continue
        allowed = valid_gap & fg
        # No fill at frame edges; the last/first rows may be clipped.
        allowed[[0, -1], :] = False
        measured = (ids > 0) & fg
        buckets = {}
        for span in range(1, min(max_px, z.shape[1]-2) + 1):
            n = z.shape[1] - span - 1
            left, right = ids[:, :n], ids[:, span+1:]
            flanks = measured[:, :n] & measured[:, span+1:] & (left != right)
            for offset in range(1, span+1):
                flanks &= allowed[:, offset:offset+n]
            rows, starts = np.nonzero(flanks)
            if not len(rows):
                continue
            za, zb = z[rows, starts], z[rows, starts+span+1]
            tol = 4*np.finfo(float).eps*np.maximum(np.maximum(np.abs(za), np.abs(zb)), depth_jump_m)
            compatible = np.abs(za-zb) <= depth_jump_m+tol
            for row, start in zip(rows[compatible], starts[compatible]):
                a, b = int(left[row, start]), int(right[row, start])
                key = (a, b)
                buckets.setdefault(key, []).append((int(row), int(start)+1, int(start)+span+1))
        for pair, runs in sorted(buckets.items()):
            # A line can contain distinct missing runs across winding lobes.
            # Build 8-connected tracks of runs, rather than concatenating
            # different gaps that happen to share a component pair.
            tracks = []
            active, current, used = {}, {}, set()
            last_row = -2
            for row, lo, hi in sorted(runs):
                if row != last_row:
                    active, current = (current if row == last_row+1 else {}), {}
                    used = set()
                    last_row = row
                # Endpoints are included here solely to let gaps that shift
                # one pixel per row (diagonal cuts) touch across adjacent rows.
                neighbors = {i for x in range(lo, hi+1) for i in active.get(x, ())}
                choices = neighbors - used
                if choices:
                    idx = min(choices)
                    tracks[idx].append((row, lo, hi))
                else:
                    idx = len(tracks)
                    tracks.append([(row, lo, hi)])
                used.add(idx)
                for x in range(lo, hi+1):
                    current.setdefault(x, []).append(idx)
            for track in tracks:
                if len(track) >= min_lines:
                    yield _Proposal(pair, axis, tuple(track))


def result_detail(status, **data):
    return dict(status=status, candidate=False, inferred_gap_pixels=0,
                scope='independent_foreground_supported_missing_depth_hypothesis', **data)
