"""Explicitly save common result debug views to an existing output directory."""
import argparse
from pathlib import Path
from example_reference_grasp import make_example
from example_surface import make_examples
from example_object_context import make_examples as object_examples


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if not args.output_dir.is_dir():parser.error('output directory must already exist')
    depth,reference,segmentation,grasp=make_example()
    for name,result in [('reference',reference),('segmentation',segmentation),('grasp',grasp)]:
        result.save_debug(args.output_dir/(name+'.png'),depth=depth)
    for name,(depth,result) in make_examples().items():
        result.save_debug(args.output_dir/('surface-'+name+'.png'),depth=depth)
    for name,(depth,result,_) in object_examples().items():
        result.save_debug(args.output_dir/('object-'+name+'.png'),depth=depth)


if __name__=='__main__':main()
