"""Observable surface evidence, including a single-sheet counterexample."""
import json
import numpy as np
from depth_image_lens import DepthSurfaceDetector,CameraIntrinsics,TablePlane


def make_examples():
    mask=np.zeros((72,96),bool);mask[5:67,5:91]=True
    k=CameraIntrinsics(150,150,48,36);plane=TablePlane((0,0,-1),2.)
    results={}
    for name in ('flat','suspended_single_sheet','step'):
        depth=np.full(mask.shape,2000,np.uint16)
        if name=='suspended_single_sheet':depth[mask]=1940
        if name=='step':depth[:,35:]=1930
        result=DepthSurfaceDetector().detect(depth,mask,intrinsics=k,table=plane,diagnostics=True)
        results[name]=(depth,result)
    return results


def main():
    print(json.dumps({k:r.to_dict() for k,(_,r) in make_examples().items()},indent=2,allow_nan=False))


if __name__=='__main__':main()
