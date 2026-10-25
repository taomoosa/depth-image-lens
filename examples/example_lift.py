"""Synthetic oblique-plane lift; no hand ROI, difference image or robot access."""
import json
import numpy as np
from depth_image_lens import CameraIntrinsics, TablePlane, LiftDetector, fit_table_plane


def main():
    intrinsics = CameraIntrinsics(100,100,39.5,29.5)
    angle = np.deg2rad(20)
    plane = TablePlane((0,-np.cos(angle),-np.sin(angle)), .8)
    # Generate an elevated surface. Calibration/mask would be caller inputs.
    rays = intrinsics.deproject(np.ones((60,80)))
    depth_m = (.12-plane.offset_m)/(rays@np.asarray(plane.normal))
    object_mask = np.zeros(depth_m.shape,bool)
    object_mask[22:55,12:68] = True
    # Only valid positive optical-axis samples are encoded; zero stays missing.
    counts = np.zeros(depth_m.shape,np.uint16)
    valid = np.isfinite(depth_m) & (depth_m>0) & (depth_m<65.535)
    counts[valid] = np.rint(depth_m[valid]*1000).astype(np.uint16)
    empty_m = -plane.offset_m/(rays@np.asarray(plane.normal))
    empty_counts = np.zeros(depth_m.shape,np.uint16)
    empty_valid = np.isfinite(empty_m) & (empty_m>0) & (empty_m<65.535)
    empty_counts[empty_valid] = np.rint(empty_m[empty_valid]*1000).astype(np.uint16)
    # In a real empty image, draw this ROI on the table only; exclude edges/clutter.
    table_mask = object_mask.copy()
    fit = fit_table_plane(empty_counts,intrinsics,table_mask,depth_scale_m=.001)
    result = LiftDetector().detect_16uc1(
        counts,intrinsics,fit.plane,object_mask,depth_scale_m=.001,
        whole_object_visible=False,
    )
    print(json.dumps({"table_fit":fit.to_dict(),"lift":result.to_dict()},indent=2,allow_nan=False))


if __name__=='__main__':main()
