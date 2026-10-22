"""python3 examples/example.py [depth.npy difference.npy config.json] -> JSON.

No arguments: a synthetic U-shaped bridge. External arrays are never unpickled.
"""
from pathlib import Path
import json
import sys
import numpy as np
from depth_image_lens import BridgeDrapeDetector


def main():
    if len(sys.argv) == 1:
        depth = np.full((100,120),2.)
        for x in range(20,101):
            top=20+int(round(35*np.sin(np.pi*(x-20)/80)))
            depth[top:top+7,x]=1.5
        difference=2.-depth
        options={"hand_rois":{"left":[14,4,26,20],"right":[94,4,106,20]}}
        params={};provenance="synthetic U bridge; not a real camera observation"
    elif len(sys.argv) == 4:
        depth=np.load(sys.argv[1],allow_pickle=False)
        difference=np.load(sys.argv[2],allow_pickle=False)
        config_path=Path(sys.argv[3]);options=json.loads(config_path.read_text())
        params=options.pop("detector",{})
        for name in ("arm_mask","work_roi","background_valid","observed_mask"):
            if name in options:
                options[name]=np.load(config_path.parent/options[name],allow_pickle=False)
        provenance="caller arrays and explicit configuration; not independently validated"
    else:
        raise SystemExit("Usage: python3 examples/example.py [depth.npy difference.npy config.json]")
    result=BridgeDrapeDetector(**params).detect(depth,difference,**options)
    print(json.dumps({"input_provenance":provenance,**result.to_dict()},indent=2,allow_nan=False))


if __name__ == "__main__": main()
