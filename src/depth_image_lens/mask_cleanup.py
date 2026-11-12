"""Opt-in geometric mask repair, never new color/depth evidence."""
from dataclasses import dataclass, field
import numpy as np
from .geometry import bool_mask


_NEIGHBORS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                   if dy or dx)


def _components8(mask):
    """NumPy-only run-length connected components, including diagonal contact."""
    labels = np.zeros(mask.shape, np.int32)
    parents = [0]

    def root(label):
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label

    previous = []
    for y, row in enumerate(mask):
        edges = np.flatnonzero(np.diff(np.r_[False, row, False]))
        current = []
        first = 0
        for start, stop in edges.reshape(-1, 2):
            label = len(parents)
            parents.append(label)
            while first < len(previous) and previous[first][1] < start:
                first += 1
            i = first
            while i < len(previous) and previous[i][0] <= stop:
                old = root(previous[i][2])
                new = root(label)
                if old != new:
                    parents[new] = old
                i += 1
            labels[y, start:stop] = label
            current.append((int(start), int(stop), label))
        previous = current
    roots = np.array([root(i) for i in range(len(parents))])
    _, lookup = np.unique(roots, return_inverse=True)
    labels = lookup[labels].astype(np.int32)
    return labels, np.bincount(labels.ravel(), minlength=int(lookup.max()) + 1)


def _box_sum(mask, radius):
    padded = np.pad(mask.astype(np.int64), radius)
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    size = 2 * radius + 1
    return (integral[size:, size:] - integral[:-size, size:]
            - integral[size:, :-size] + integral[:-size, :-size])


def _neighbor_component_count(patch, value):
    remaining = {(y, x) for y in range(3) for x in range(3)
                 if (y, x) != (1, 1) and patch[y, x] == value}
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            y, x = stack.pop()
            for dy, dx in _NEIGHBORS:
                neighbor = (y + dy, x + dx)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


@dataclass(frozen=True)
class MaskCleanupResult:
    """Separate snapshots; corrections are geometric hypotheses, not observations."""
    original_mask: np.ndarray = field(repr=False)
    object_mask: np.ndarray = field(repr=False)
    observed_mask: np.ndarray = field(repr=False)
    protected_mask: np.ndarray = field(repr=False)
    added_mask: np.ndarray = field(repr=False)
    removed_mask: np.ndarray = field(repr=False)
    filled_hole_mask: np.ndarray = field(repr=False)
    smoothed_added_mask: np.ndarray = field(repr=False)
    smoothed_removed_mask: np.ndarray = field(repr=False)
    parameters: dict
    metrics: dict
    warnings: list
    # Optional trailing field preserves construction with the original arguments.
    # cleanup_mask always returns a boolean array here, even when closing is off.
    closed_gap_mask: np.ndarray = field(default=None, repr=False)

    @property
    def changed_mask(self):
        return self.added_mask | self.removed_mask

    def to_dict(self):
        """JSON-compatible settings and pixel counts, without image arrays."""
        return dict(parameters=dict(self.parameters), metrics=dict(self.metrics),
                    warnings=list(self.warnings))


def cleanup_mask(object_mask, *, max_hole_area=0, smoothing_radius_px=0,
                 closing_radius_px=0, max_gap_area=0,
                 observed_mask=None, protected_mask=None):
    """Fill small enclosed holes and/or smooth the exterior of a bool HxW mask.

    All operations default to off. Gap closing requires both closing_radius_px
    (0..5) and max_gap_area (pixels) to be positive. It proposes square binary
    closing additions, then accepts whole candidate components within the area
    limit only if they touch one existing foreground component and avoid the
    frame/unknown/protected 2*radius halo and original enclosed-hole rim.
    This can seal a real open notch; it is an explicitly geometric hypothesis.

    Hole filling and smoothing default to off. Hole area is in pixels, inclusive of the
    limit; smoothing is one square-window majority pass, radius 0..5 pixels.
    Input arrays are never modified. Unknown (unobserved) and protected pixels
    retain their input values, including foreground values. Pass deferred,
    suppressed, excluded or uncertain regions as protected when appropriate.

    Background and foreground use conservative 8-connectivity. Holes touching
    the frame, unknown/protected pixels or their one-pixel rim are not filled.
    Filling cannot join separate foreground components. Smoothing preserves
    enclosed holes and their rim, freezes a radius-wide frame/unknown/protected
    halo, and applies sequential topology-guarded changes in row-major order.
    Smoothing does not erase, split or merge components or close open gaps.
    Optional gap closing can change background topology but never joins separate
    foreground components. Newly closed pockets are not hole-filled in this call.
    Real holes, corners and thin appendages cannot be identified from a mask:
    small real holes may be filled and optional smoothing can round/shorten
    real details. No depth values or source evidence are changed.
    """
    source = np.asarray(object_mask)
    if source.dtype != bool or source.ndim != 2 or min(source.shape) < 1:
        raise ValueError('object_mask must be a nonempty bool HxW array')
    if type(max_hole_area) is not int or max_hole_area < 0:
        raise ValueError('max_hole_area must be a nonnegative integer')
    if type(smoothing_radius_px) is not int or not 0 <= smoothing_radius_px <= 5:
        raise ValueError('smoothing_radius_px must be an integer in 0..5')
    if type(closing_radius_px) is not int or not 0 <= closing_radius_px <= 5:
        raise ValueError('closing_radius_px must be an integer in 0..5')
    if type(max_gap_area) is not int or max_gap_area < 0:
        raise ValueError('max_gap_area must be a nonnegative integer')
    shape = source.shape
    original = source.copy()
    cleaned = source.copy()
    observed = (np.ones(shape, bool) if observed_mask is None else
                bool_mask(observed_mask, shape, 'observed_mask').copy())
    protected = (np.zeros(shape, bool) if protected_mask is None else
                 bool_mask(protected_mask, shape, 'protected_mask').copy())
    blocked = ~observed | protected
    filled = np.zeros(shape, bool)
    enclosed = np.zeros(shape, bool)
    hole_count = None
    filled_count = topology_rejected = proposed_count = 0

    if max_hole_area or smoothing_radius_px or (closing_radius_px and max_gap_area):
        background_labels, areas = _components8(~original)
        exterior = np.zeros(len(areas), bool)
        exterior[0] = True
        exterior[np.unique(np.r_[background_labels[0], background_labels[-1],
                                 background_labels[:, 0], background_labels[:, -1]])] = True
        enclosed = ~exterior[background_labels]
        hole_ids = np.flatnonzero(~exterior)
        hole_count = len(hole_ids)
        if max_hole_area:
            foreground_labels, _ = _components8(original)
            blocked_rim = _box_sum(blocked, 1) > 0
            blocked_counts = np.bincount(background_labels[blocked_rim], minlength=len(areas))
            candidates = ~exterior & (areas <= max_hole_area) & (blocked_counts == 0)
            # Collect neighboring foreground labels in one image pass per
            # direction, not one full-image scan for every pinhole.
            sentinel = int(foreground_labels.max()) + 1
            low = np.full(len(areas), sentinel, np.int32)
            high = np.zeros(len(areas), np.int32)
            height, width = shape
            if candidates.any():
                for dy, dx in _NEIGHBORS:
                    y0, y1 = max(0, -dy), min(height, height - dy)
                    x0, x1 = max(0, -dx), min(width, width - dx)
                    bg = background_labels[y0:y1, x0:x1]
                    fg = foreground_labels[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
                    selected = candidates[bg] & (fg > 0)
                    np.minimum.at(low, bg[selected], fg[selected])
                    np.maximum.at(high, bg[selected], fg[selected])
            # A hole containing an independent island must not join it to
            # the surrounding object, even when its background area is small.
            fill_labels = candidates & (low == high) & (high > 0)
            filled = fill_labels[background_labels]
            filled_count = int(fill_labels.sum())
            cleaned |= filled

    closed_gap = np.zeros(shape, bool)
    gap_candidates = gap_rejected = gap_count = 0
    if closing_radius_px and max_gap_area:
        radius = closing_radius_px
        size = 2 * radius + 1
        # Closing is only a proposal. Do not clip unsafe pieces off a candidate:
        # reject the complete addition component instead.
        dilated = _box_sum(cleaned, radius) > 0
        closed = _box_sum(dilated, radius) == size * size
        proposal = closed & ~cleaned
        gap_labels, gap_areas = _components8(proposal)
        gap_candidates = len(gap_areas) - 1
        unsafe = (_box_sum(blocked, 2 * radius) > 0) | (_box_sum(enclosed, 1) > 0)
        halo = 2 * radius
        unsafe[:halo] = unsafe[-halo:] = True
        unsafe[:, :halo] = unsafe[:, -halo:] = True
        unsafe_counts = np.bincount(gap_labels[unsafe], minlength=len(gap_areas))
        candidates = (gap_areas <= max_gap_area) & (unsafe_counts == 0)
        candidates[0] = False
        fg_labels, _ = _components8(cleaned)
        sentinel = int(fg_labels.max()) + 1
        low = np.full(len(gap_areas), sentinel, np.int32)
        high = np.zeros(len(gap_areas), np.int32)
        height, width = shape
        for dy, dx in _NEIGHBORS:
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            bg = gap_labels[y0:y1, x0:x1]
            fg = fg_labels[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
            selected = candidates[bg] & (fg > 0)
            np.minimum.at(low, bg[selected], fg[selected])
            np.maximum.at(high, bg[selected], fg[selected])
        accepted = candidates & (low == high) & (high > 0)
        closed_gap = accepted[gap_labels]
        gap_count = int(accepted.sum())
        gap_rejected = gap_candidates - gap_count
        cleaned |= closed_gap

    before_smoothing = cleaned.copy()
    radius = smoothing_radius_px
    if radius:
        safe = _box_sum(blocked, radius) == 0
        safe &= _box_sum(enclosed | closed_gap, 1) == 0
        # No padding value is evidence beyond the image or observed domain.
        safe[:radius] = safe[-radius:] = False
        safe[:, :radius] = safe[:, -radius:] = False
        majority = _box_sum(cleaned, radius) > ((2 * radius + 1) ** 2 // 2)
        proposals = safe & (majority != cleaned)
        proposed_count = int(proposals.sum())
        for y, x in np.argwhere(proposals):
            patch = cleaned[y - 1:y + 2, x - 1:x + 2]
            # Check against the evolving mask. Parallel independent tests can
            # together split a thin neck or join two initially separate objects.
            if (_neighbor_component_count(patch, True) == 1
                    and _neighbor_component_count(patch, False) == 1):
                cleaned[y, x] = majority[y, x]
            else:
                topology_rejected += 1

    added = cleaned & ~original
    removed = original & ~cleaned
    smoothed_added = cleaned & ~before_smoothing
    smoothed_removed = before_smoothing & ~cleaned
    metrics = dict(original_pixels=int(original.sum()), output_pixels=int(cleaned.sum()),
                   added_pixels=int(added.sum()), removed_pixels=int(removed.sum()),
                   changed_pixels=int((added | removed).sum()),
                   enclosed_holes=hole_count, filled_holes=filled_count,
                   filled_hole_pixels=int(filled.sum()),
                   closed_gap_pixels=int(closed_gap.sum()), closed_gaps=gap_count,
                   closing_candidate_components=gap_candidates,
                   closing_rejected_components=gap_rejected,
                   smoothing_added_pixels=int(smoothed_added.sum()),
                   smoothing_removed_pixels=int(smoothed_removed.sum()),
                   smoothing_proposed_pixels=proposed_count,
                   smoothing_topology_rejected_pixels=topology_rejected,
                   protected_pixels=int(protected.sum()), unobserved_pixels=int((~observed).sum()))
    parameters = dict(max_hole_area=max_hole_area, smoothing_radius_px=radius,
                      closing_radius_px=closing_radius_px, max_gap_area=max_gap_area,
                      connectivity=8, smoothing_passes=1 if radius else 0,
                      smoothing_order='row_major_sequential_topology_guard')
    warnings = ['geometric_corrections_are_not_observed_color_or_depth_evidence',
                'small_real_holes_and_segmentation_holes_are_indistinguishable',
                'smoothing_can_round_corners_and_shorten_thin_appendages',
                ('closing_can_seal_real_notches_and_create_enclosed_pockets'
                 if closing_radius_px and max_gap_area else
                 'open_gaps_and_missing_objects_are_not_reconstructed'),
                'pixel_settings_depend_on_image_resolution',
                'real_image_accuracy_unvalidated']
    return MaskCleanupResult(original, cleaned, observed, protected, added, removed,
                             filled, smoothed_added, smoothed_removed,
                             parameters, metrics, warnings, closed_gap_mask=closed_gap)
