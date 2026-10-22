"""No args: synthetic U. Or depth.npy difference.npy config.auto.json."""
import sys,json
from pathlib import Path
import numpy as np
from depth_image_lens import BridgeDrapeDetector

if len(sys.argv)==1:
    depth=np.full((100,120),2.)
    for x in range(20,101):
        y=20+round(35*np.sin(np.pi*(x-20)/80));depth[y:y+7,x]=1.5
    difference=2.-depth;config={'upper_band':[8,20]}
elif len(sys.argv)==4:
    depth=np.load(sys.argv[1],allow_pickle=False);difference=np.load(sys.argv[2],allow_pickle=False)
    path=Path(sys.argv[3]);config=json.loads(path.read_text())
    for k in ('arm_mask','work_roi','background_valid','observed_mask'):
        if k in config:config[k]=np.load(path.parent/config[k],allow_pickle=False)
else:raise SystemExit('Usage: python3 examples/example_auto.py [depth.npy difference.npy config.auto.json]')
detector=BridgeDrapeDetector(**config.pop('detector',{}))
print(json.dumps(detector.detect_auto(depth,difference,**config).to_dict(),indent=2,allow_nan=False))
