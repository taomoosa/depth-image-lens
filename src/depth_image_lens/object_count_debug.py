"""Optional group-aware visualization; no changes to counting or its evidence."""
import colorsys
import numpy as np
from .debug import DebugResultMixin, _opencv, depth_canvas


def _group_color(group_id):
    """An RGB display color keyed only by ID, including beyond the first palette."""
    first = ((50, 165, 245), (255, 190, 45), (170, 100, 230), (40, 205, 140),
             (245, 95, 120), (50, 210, 220), (230, 135, 40), (170, 205, 45))
    if group_id <= len(first):
        return first[group_id - 1]
    # Never cycle a short palette. IDs are also printed because no palette can
    # make 128 small adjacent groups reliably distinguishable by color alone.
    rgb = colorsys.hsv_to_rgb((group_id * .618033988749895) % 1,
                             (.65, .85, .95)[group_id % 3],
                             (.85, 1.)[group_id % 2])
    return tuple(round(channel * 255) for channel in rgb)


def _group_images(result, source, alpha):
    """Full-resolution RGB images: tint over source and an untinted ID map."""
    colors = np.array([(35, 35, 35)] + [_group_color(i)
                      for i in range(1, result.candidate_count + 1)], np.uint8)
    classified = colors[result.label_map]
    overlay = source.copy()
    assigned = result.label_map > 0
    overlay[assigned] = np.rint((1-alpha)*overlay[assigned]
                              + alpha*classified[assigned]).astype(np.uint8)
    return overlay, classified


def _label_groups(image, labels, group_ids):
    """Label visible fragments at interior pixels; skip colliding badges."""
    cv = _opencv()
    occupied = []
    labeled = set()
    h, w = labels.shape
    for group_id in group_ids:
        n, components, stats, _ = cv.connectedComponentsWithStats(
            (labels == group_id).astype(np.uint8), connectivity=8)
        for component in sorted(range(1, n), key=lambda i: -stats[i, cv.CC_STAT_AREA]):
            x, y, cw, ch, _ = stats[component]
            mask = components[y:y+ch, x:x+cw] == component
            # Padding makes edge-touching and fully filled components well-defined.
            distance = cv.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv.DIST_L2, 3)[1:-1, 1:-1]
            cy, cx = np.unravel_index(int(distance.argmax()), distance.shape)
            text = f'G{group_id}'
            (tw, th), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, .4, 1)
            bw, bh = tw+8, th+baseline+6
            if bw > w or bh > h:
                continue
            left = min(max(0, int(x+cx)-bw//2), w-bw)
            top = min(max(0, int(y+cy)-bh//2), h-bh)
            box = (left, top, left+bw, top+bh)
            if any(left < r+2 and left+bw+2 > l and top < b+2 and top+bh+2 > t
                   for l, t, r, b in occupied):
                continue
            cv.rectangle(image, (left, top), (left+bw-1, top+bh-1), (20, 20, 20), -1)
            cv.rectangle(image, (left, top), (left+bw-1, top+bh-1), _group_color(group_id), 1)
            cv.putText(image, text, (left+4, top+th+3), cv.FONT_HERSHEY_SIMPLEX,
                       .4, (255, 255, 255), 1, cv.LINE_AA)
            occupied.append(box)
            labeled.add(group_id)
    return len(labeled)


def render_object_count_debug(result, color, *, color_order, depth, depth_scale_m,
                              mask_alpha, panel, max_lines, group_labels):
    """Prepend the group view while preserving the shared mask/save contracts."""
    if type(group_labels) is not bool:
        raise ValueError('group_labels must be bool')
    labels = result.label_map
    if (not isinstance(labels, np.ndarray) or labels.ndim != 2
            or labels.dtype.kind not in 'iu' or min(labels.shape) < 2
            or labels.shape != result.object_mask.shape
            or (labels < 0).any() or (labels > result.candidate_count).any()):
        raise ValueError('label_map must match the result grid and contain IDs 0..candidate_count')
    # This call validates the existing render options and every recorded mask,
    # and retains separate uncertainty/rejected/excluded/observed evidence views.
    evidence = DebugResultMixin.render_debug(
        result, color, color_order=color_order, depth=depth, depth_scale_m=depth_scale_m,
        mask_alpha=mask_alpha, panel=panel, max_lines=max_lines)
    cv = _opencv()
    if color is not None:
        color = np.asarray(color)
        source = np.array(color if color_order == 'RGB' else color[..., ::-1], copy=True)
    elif depth is not None:
        source = depth_canvas(depth, depth_scale_m)
    else:
        source = np.full((*labels.shape, 3), 35, np.uint8)
    overlay, classified = _group_images(result, source, mask_alpha)
    # Enlarge small scenes for legible IDs, never interpolate group identities.
    h, w = labels.shape
    scale = min(480/w, 480/h, (evidence.shape[1]//2-16)/w)
    tw, th = max(2, round(w*scale)), max(2, round(h*scale))
    overlay = cv.resize(overlay, (tw, th), interpolation=cv.INTER_NEAREST)
    classified = cv.resize(classified, (tw, th), interpolation=cv.INTER_NEAREST)
    displayed_labels = cv.resize(labels.astype(np.int32), (tw, th), interpolation=cv.INTER_NEAREST)
    ids = range(1, result.candidate_count+1)
    labeled = _label_groups(overlay, displayed_labels, ids) if group_labels else 0

    width = evidence.shape[1]
    columns = max(1, width//280)
    cell_width = width//columns
    rows = max(1, (result.candidate_count+columns-1)//columns)
    legend_y = 96+th
    strip = np.full((legend_y+rows*24+72, width, 3), 24, np.uint8)

    def write(text, x, y, ink=(235, 235, 235)):
        cv.putText(strip, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, .4, ink, 1, cv.LINE_AA)

    count = 'None (unresolved)' if result.count is None else str(result.count)
    write(f'count: {count} | candidate_count: {result.candidate_count} | state: {result.state}', 8, 20)
    write('Visible color-group hypotheses, not verified physical objects. Colors/IDs are not tracked across frames.', 8, 39)
    for i, (title, tile) in enumerate((('Groups over source (G = group ID)', overlay),
                                     ('Group map (display colors, not measured RGB)', classified))):
        x = i*(width//2)+(width//2-tw)//2
        write(title, i*(width//2)+8, 62)
        strip[72:72+th, x:x+tw] = tile

    # Unlike the bounded scalar panel, this legend includes every retained ID.
    # Diagnostics are not needed: the core result always contains this evidence.
    for i, group in enumerate(sorted(result.groups, key=lambda group: group['id'])):
        group_id = group['id']
        x = (i % columns)*cell_width+8
        y = legend_y+(i//columns)*24
        cv.rectangle(strip, (x, y-12), (x+16, y), _group_color(group_id), -1)
        uncertain = bool(group['reasons']) or bool(result.uncertain_mask[labels == group_id].any())
        write(f"G{group_id}{'*' if uncertain else ''}: {group['area_px']} px / {group['fragment_count']} parts", x+24, y)
    if not result.groups:
        write('No retained groups. An unresolved count is not a verified zero.', 8, legend_y)
    footer = legend_y+rows*24
    write('0 / dark gray = no retained group. See separate evidence masks below for rejections, uncertainty and exclusions.', 8, footer)
    write('* = group reasons or overlapping uncertainty. Display colors do not indicate confidence.', 8, footer+19)
    write((f'ID badges shown for {labeled}/{result.candidate_count} groups; tiny/crowded fragments may omit badges.'
           if group_labels else 'Spatial ID badges disabled (group_labels=False); all group IDs remain in the legend.')
          + ' Views are resized.', 8, footer+38)
    if color_order == 'BGR':
        strip = np.ascontiguousarray(strip[..., ::-1])
    return np.concatenate((strip, evidence), axis=0)
