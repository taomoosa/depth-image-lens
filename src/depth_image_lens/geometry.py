"""Rectified optical-axis depth geometry, in the depth-camera coordinate frame."""
from dataclasses import dataclass
import numpy as np


def bool_mask(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != bool:
        raise ValueError(f"{name} must be a boolean array of shape {shape}")
    return value


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all() or min(self.fx, self.fy) <= 0:
            raise ValueError("finite intrinsics with positive focal lengths required")

    def deproject(self, depth_m):
        """HxW optical-axis Z in metres -> HxWx3. No distortion correction."""
        z = np.asarray(depth_m, float)
        if z.ndim != 2:
            raise ValueError("depth must be HxW optical-axis Z in metres")
        y, x = np.indices(z.shape)
        return np.stack(((x-self.cx)*z/self.fx, (y-self.cy)*z/self.fy, z), axis=-1)

    def cropped(self, x0, y0):
        """Intrinsics for an unscaled crop starting at (x0,y0)."""
        return CameraIntrinsics(self.fx, self.fy, self.cx-x0, self.cy-y0)


@dataclass(frozen=True)
class TablePlane:
    """normal.dot(point)+offset_m; positive on the object's above-table side.

    Camera movement invalidates a stored plane. Estimate from a current table
    ROI or supply a current plane. A floor/wall ROI cannot be recognized here.
    """
    normal: tuple[float, float, float]
    offset_m: float

    def __post_init__(self):
        n = np.asarray(self.normal, float)
        if n.shape != (3,) or not np.isfinite(n).all() or not np.isfinite(self.offset_m):
            raise ValueError("finite plane normal and offset required")
        length = np.linalg.norm(n)
        if not np.isfinite(length) or length < 1e-12:
            raise ValueError("finite nonzero plane normal length required")
        if not np.isfinite(self.offset_m/length):
            raise ValueError("normalized plane offset must be finite")
        object.__setattr__(self, "normal", tuple(float(x) for x in n/length))
        object.__setattr__(self, "offset_m", float(self.offset_m/length))

    def height(self, points_m):
        return np.asarray(points_m, float) @ np.asarray(self.normal) + self.offset_m



def largest_component(mask):
    """Four-connected pixel area. No physical-area claim or hole filling."""
    remaining = mask.copy()
    largest = 0
    for y, x in zip(*np.nonzero(mask)):
        if not remaining[y, x]:
            continue
        remaining[y, x] = False
        stack = [(y, x)]; size = 0
        while stack:
            a, b = stack.pop(); size += 1
            for c, d in ((a-1,b),(a+1,b),(a,b-1),(a,b+1)):
                if 0 <= c < mask.shape[0] and 0 <= d < mask.shape[1] and remaining[c,d]:
                    remaining[c,d] = False; stack.append((c,d))
        largest = max(largest, size)
    return largest
