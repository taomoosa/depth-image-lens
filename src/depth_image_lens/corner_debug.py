"""Optional rendering of recorded CornerResult evidence, without detector changes."""
from pathlib import Path
import numpy as np
from .geometry import bool_mask
from .diagnostics import _atomic_write


# RGB display colors; accepted is a heuristic candidate, not grasp assurance.
_STATUS_COLORS = {"accepted": (30, 210, 90), "suspected": (245, 90, 70),
                  "unknown": (255, 195, 40)}


def render_corner_debug(result, color, *, color_order, mask_alpha=.2,
                        show_contours=True, max_labels=12, panel=True):
    from .corner import _cv2
    cv2 = _cv2()
    raw = np.asarray(color)
    if color_order not in ("RGB", "BGR"):
        raise ValueError("explicit RGB or BGR color_order required")
    if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[2] != 3 or min(raw.shape[:2]) < 2:
        raise ValueError("source color must be uint8 HxWx3")
    mask = bool_mask(result.object_mask, raw.shape[:2], "result.object_mask")
    if (not np.isscalar(mask_alpha) or isinstance(mask_alpha, (str, bool))
            or not np.isfinite(mask_alpha) or not 0 <= mask_alpha <= 1):
        raise ValueError("mask_alpha must be finite in [0,1]")
    if type(show_contours) is not bool or type(panel) is not bool:
        raise ValueError("show_contours and panel must be bool")
    if type(max_labels) is not int or not 0 <= max_labels <= 100:
        raise ValueError("max_labels must be an integer in [0,100]")
    rgb = np.array(raw if color_order == "RGB" else raw[..., ::-1], copy=True, order="C")
    rgb[mask] = np.rint((1-mask_alpha)*rgb[mask] + mask_alpha*np.array([45,180,220])).astype(np.uint8)
    contours = result.diagnostics.get("contours_xy")
    if show_contours and contours is not None:
        for contour in contours:
            points = np.asarray(contour, float)
            if points.ndim == 2 and points.shape[1] == 2 and len(points) and np.isfinite(points).all():
                # Recorded detector contours are in-frame; clip only drawing coordinates.
                points = np.clip(points, [0,0], [raw.shape[1]-1,raw.shape[0]-1])
                cv2.polylines(rgb, [np.rint(points).astype(np.int32)], True, (70,220,245), 1, cv2.LINE_AA)
    h, w = rgb.shape[:2]
    rows = []; skipped = 0
    for i, candidate in enumerate(result.candidates):
        xy = np.asarray(candidate.get("xy", []), float)
        status = candidate.get("status", "unknown")
        ink = _STATUS_COLORS.get(status, _STATUS_COLORS["unknown"])
        drawable = xy.shape == (2,) and np.isfinite(xy).all() and 0 <= xy[0] < w and 0 <= xy[1] < h
        if drawable:
            x,y = np.rint(xy).astype(int); x=min(x,w-1);y=min(y,h-1)
            cv2.circle(rgb, (x,y), 5, (0,0,0), 3, cv2.LINE_AA)
            cv2.circle(rgb, (x,y), 4, ink, 2, cv2.LINE_AA)
            if i < max_labels:
                label = str(i)
                pos = (max(0,min(x+6,w-16)), max(12,min(y-5,h-2)))
                cv2.putText(rgb,label,pos,cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(rgb,label,pos,cv2.FONT_HERSHEY_SIMPLEX,.4,ink,1,cv2.LINE_AA)
        else: skipped += 1
        if i < max_labels:
            evidence = candidate.get("evidence", {})
            metrics = []
            for key, name, scale, unit in (("angle_deg","angle",1,"deg"),
                    ("line_residual_px","line",1,"px"), ("local_residual_p95_m","depth",1000,"mm")):
                value = candidate.get(key) if key == "angle_deg" else evidence.get(key)
                if isinstance(value,(int,float,np.number)) and np.isfinite(value):
                    metrics.append(f"{name}={value*scale:.2f}{unit}")
            reasons = "; ".join(str(x) for x in candidate.get("reasons", [])) or "no rejection recorded"
            warnings = "; ".join(str(x) for x in candidate.get("warnings", []))
            position = f"({xy[0]:.1f},{xy[1]:.1f})" if xy.shape==(2,) and np.isfinite(xy).all() else "unavailable"
            rows.append((ink, [f"#{i} {status} {position}", " | ".join(metrics) or "measurements unavailable",
                reasons, warnings] if warnings else [f"#{i} {status} {position}",
                " | ".join(metrics) or "measurements unavailable", reasons]))
    if panel:
        import textwrap
        lines = [("Corner candidates (not grasp assurance)",(235,235,235))]
        for status, ink in _STATUS_COLORS.items():
            count = sum(c.get("status") == status for c in result.candidates)
            lines.append((f"{status}: {count}", ink))
        lines += [("Cyan: supplied mask / recorded contour",(150,220,235)),
            ("IDs: zero-based result.candidates indices",(210,210,210)),
            (f"Mode: {result.summary.get('mode', 'unspecified')}; missing metrics omitted.",(210,210,210)),
            ("Only returned candidates are drawn.",(210,210,210)),
            ("Pre-proposal filters / merged points not reconstructed.",(210,210,210))]
        if contours is None: lines.append(("Contours unavailable: diagnostics not recorded.",(255,195,40)))
        elif not show_contours: lines.append(("Recorded contour drawing disabled.",(210,210,210)))
        if not result.candidates: lines.append(("No returned candidates; not proof of no physical corners.",(255,195,40)))
        if skipped: lines.append((f"{skipped} unavailable/off-image positions not drawn.",(255,195,40)))
        if len(result.candidates)>max_labels:
            lines.append((f"Details limited to first {max_labels}; all in-frame markers drawn.",(255,195,40)))
        for ink, texts in rows:
            lines.append(("",ink))
            for text in texts:
                # Render text only, with bounded wrapping; never interpret paths or markup.
                lines.extend((part,ink) for part in textwrap.wrap(text,62)[:3])
        canvas = np.full((max(h,24+18*len(lines)),w+520,3),24,np.uint8)
        canvas[:h,:w] = rgb
        for i,(text,ink) in enumerate(lines):
            cv2.putText(canvas,text,(w+12,22+i*18),cv2.FONT_HERSHEY_SIMPLEX,.43,ink,1,cv2.LINE_AA)
        rgb = canvas
    return rgb if color_order == "RGB" else np.ascontiguousarray(rgb[..., ::-1])


def save_corner_debug(result, path, color, *, color_order, overwrite=False, **options):
    from .corner import _cv2
    if type(overwrite) is not bool: raise ValueError("overwrite must be bool")
    destination = Path(path)
    if destination.suffix.lower() != ".png": raise ValueError("debug image destination must end in .png")
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise ValueError("debug image parent must be an existing non-symlink directory")
    image = render_corner_debug(result,color,color_order=color_order,**options)
    bgr = image[..., ::-1] if color_order == "RGB" else image
    ok, encoded = _cv2().imencode(".png", bgr)
    if not ok: raise OSError("OpenCV could not encode the debug PNG")
    _atomic_write(destination, lambda stream: stream.write(encoded.tobytes()), overwrite=overwrite)
    return destination
