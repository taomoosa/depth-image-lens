"""Optional local diagnostic persistence; no additional image dependencies."""
import json
import os
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np


def mask_path(mask):
    """Encode each horizontal run; do not interpolate or fill missing pixels."""
    runs = []
    for y, row in enumerate(mask):
        edges = np.diff(np.r_[False, row, False].astype(int))
        for x0, x1 in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            runs.append(f"M{x0},{y}h{x1-x0}v1h-{x1-x0}z")
    return "".join(runs)


def _atomic_write(destination, writer, *, overwrite=True):
    """Replace one directory entry without opening its existing symlink target."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".depth-debug-", dir=destination.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Publish a new directory entry atomically; existing files/links fail.
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_debug(result, output):
    output = Path(output)
    if output.is_symlink():
        raise ValueError("output directory must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    arrays = {name: getattr(result, name) for name in ("foreground_mask", "work_mask", "invalid_mask")}
    arrays.update({name: value for name, value in result.diagnostics.items() if isinstance(value, np.ndarray)})
    _atomic_write(output / "arrays.npz", lambda stream: np.savez_compressed(stream, **arrays))
    payload = (json.dumps(result.to_dict(), indent=2, allow_nan=False) + "\n").encode("utf-8")
    _atomic_write(output / "result.json", lambda stream: stream.write(payload))
    h, w = result.foreground_mask.shape
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {2*w} {2*(h+20)}">',
           '<rect width="100%" height="100%" fill="white"/>']
    panels = [
        ("Foreground / invalid", [("foreground_mask", "#777"), ("invalid_mask", "#d33")]),
        ("Search / bridge / anchors", [("candidate_search_mask", "#e7f1fa"), ("search_mask", "#dbeafa"),
         ("opposite_search_mask", "#dbbaee"), ("bridge_corridor_mask", "#389958"),
         ("left_anchor_mask", "#e58d22"), ("right_anchor_mask", "#387de0"),
         ("inferred_gap_mask", "#e6ce28"), ("depth_rejected_mask", "#bd5c91")]),
        (f"Sag path (used: {result.diagnostics.get('sag_path_used', False)})", [("foreground_mask", "#ddd")]),
        ("Anchor-connected drape", [("left_drape_component_mask", "#e58d22"), ("right_drape_component_mask", "#387de0")]),
    ]
    for i, (title, layers) in enumerate(panels):
        x, y = (i % 2)*w, (i // 2)*(h+20)
        svg.append(f'<g transform="translate({x},{y})"><text x="2" y="12" font-size="9">{escape(title)}</text><g transform="translate(0,20)">')
        for name, color in layers:
            if name in arrays:
                svg.append(f'<path d="{mask_path(arrays[name])}" fill="{color}" opacity="0.75"/>')
        if i == 1:
            for box in result.auto.get("hand_roi_candidates", {}).values():
                x0,y0,x1,y1 = box
                svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="#111" stroke-width="1"/>')
        if i == 2:
            for name,color in [("sag_chord_xy", "#387de0"), ("sag_path_xy", "#d66b00")]:
                points = arrays.get(name, np.empty((0,2)))
                # Separate points preserve missing bins; do not draw false bridges across gaps.
                if name == "sag_path_xy":
                    for px,py in points:svg.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="0.6" fill="{color}"/>')
                else:
                    coords=" ".join(f"{px:.3f},{py:.3f}" for px,py in points)
                    svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="0.8"/>')
        svg.append('</g></g>')
    svg.append('</svg>')
    payload = "\n".join(svg).encode("utf-8")
    _atomic_write(output / "overlay.svg", lambda stream: stream.write(payload))
