"""Save diagnostic arrays, JSON and an SVG without image-library dependencies.

Run from the repository root after installation:
  python examples/example_debug.py --output debug-output
  python examples/example_debug.py --depth depth.npy --difference difference.npy \
      --config examples/config.auto.json --output debug-output
"""
import argparse
import json
from pathlib import Path
import numpy as np
from depth_image_lens import BridgeDrapeDetector
from depth_image_lens.diagnostics import save_debug



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depth');parser.add_argument('--difference');parser.add_argument('--config')
    parser.add_argument('--output',default='debug-output')
    args=parser.parse_args()
    if not any((args.depth,args.difference,args.config)):
        depth=np.full((100,120),2.)
        for x in range(20,101):
            top=20+round(35*np.sin(np.pi*(x-20)/80));depth[top:top+7,x]=1.5
        difference=2.-depth;config={'upper_band':[8,20]}
    elif all((args.depth,args.difference,args.config)):
        depth=np.load(args.depth,allow_pickle=False);difference=np.load(args.difference,allow_pickle=False)
        path=Path(args.config);config=json.loads(path.read_text())
        for k in ('arm_mask','work_roi','background_valid','observed_mask'):
            if k in config:config[k]=np.load(path.parent/config[k],allow_pickle=False)
    else:parser.error('Provide depth, difference and config together, or none for the synthetic example.')
    config.pop('diagnostics',None)
    result=BridgeDrapeDetector(**config.pop('detector',{})).detect_auto(depth,difference,diagnostics=True,**config)
    save_debug(result,args.output)
    print(json.dumps({'bridge':result.bridge,'large_sag':result.large_sag,'drape':result.drape,'output':str(Path(args.output))},allow_nan=False))


if __name__=='__main__':main()
