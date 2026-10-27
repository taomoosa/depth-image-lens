"""Public spatial results share explicit, nonmutating optional debug output."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from depth_image_lens import (BridgeDrapeDetector,IndependentGraspDetector,
    ObjectOnlyGraspDetector,LiftDetector,CameraIntrinsics,TablePlane,fit_table_plane,
    SeededColorAssistant,ReferenceGripperSegmenter,DepthSurfaceDetector)
from .test_bridge_drape import scene,inputs,ROIS
from .test_reference import reference_scene
from .rgbd_cases import cases
try:
    import cv2
except ImportError:
    cv2=None


@unittest.skipIf(cv2 is None,'optional OpenCV debug extra not installed')
class CommonDebugTests(unittest.TestCase):
    def results(self):
        z,diff=inputs(scene());m=scene();hands={k:np.zeros_like(m) for k in ('left','right')}
        hands['left'][4:20,14:26]=True;hands['right'][4:20,94:106]=True
        k=CameraIntrinsics(150,150,60,50);t=TablePlane((0,0,-1),2.)
        yield BridgeDrapeDetector().detect(z,diff,ROIS,diagnostics=True),z
        yield IndependentGraspDetector().detect(z,m,hands,depth_unit='m',registered_to_depth=True,diagnostics=True),z
        yield ObjectOnlyGraspDetector().detect(z,m,depth_unit='m',difference=diff,diagnostics=True),z
        yield LiftDetector().detect(z,k,t,m,diagnostics=True),z
        yield fit_table_plane(np.full(z.shape,2.),k,np.ones_like(m),depth_scale_m=1,diagnostics=True),z
        c=next(cases());res=SeededColorAssistant().segment_16uc1(c['depth'],c['difference'],c['color'],
            object_seeds=c['object_seeds'],hand_seeds=c['hand_seeds'],color_order='RGB',registered_to_depth=True)
        yield res,c['depth']*.001
        e,g,rgb,static,rois,_,_=reference_scene();seg=ReferenceGripperSegmenter()
        ref=seg.calibrate(e,g,rois,static,pose_key='A',registered_to_depth=True)
        yield ref,g*.001
        yield seg.segment(g,ref,pose_key='A',registered_to_depth=True,diagnostics=True),g*.001
        yield DepthSurfaceDetector().detect(z,m,depth_scale_m=1,intrinsics=k,table=t,diagnostics=True),z

    def test_all_result_views_json_and_nonmutation(self):
        for result,z in self.results():
            with self.subTest(result=type(result).__name__):
                before=json.dumps(result.to_dict(),allow_nan=False,sort_keys=True)
                old=z.copy();rgb=np.zeros((*z.shape,3),np.uint8);rgb[:]=(210,30,70)
                out=result.render_debug(rgb,color_order='RGB');self.assertEqual(out.dtype,np.uint8)
                np.testing.assert_array_equal(result.render_debug(rgb[...,::-1],color_order='BGR'),out[...,::-1])
                gray=result.render_debug(depth=z,depth_scale_m=1);self.assertEqual(gray.ndim,3)
                np.testing.assert_array_equal(z,old);self.assertTrue((rgb==(210,30,70)).all())
                self.assertEqual(before,json.dumps(result.to_dict(),allow_nan=False,sort_keys=True))

    def test_diagnostics_off_and_empty(self):
        z=np.ones((30,40));m=np.zeros(z.shape,bool);k=CameraIntrinsics(100,100,20,15);t=TablePlane((0,0,-1),1)
        a=LiftDetector().detect(z,k,t,m)
        self.assertFalse(a.diagnostics)
        with self.assertRaises(ValueError):a.render_debug()
        self.assertGreater(a.render_debug(depth=z,depth_scale_m=1).size,0)
        b=DepthSurfaceDetector().detect(z,m,depth_scale_m=1)
        self.assertGreater(b.render_debug().size,0)

    def test_explicit_save_readback_and_no_overwrite(self):
        result,z=next(self.results());rgb=np.zeros((*z.shape,3),np.uint8);rgb[:]=(200,20,90)
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/'debug.png';result.save_debug(dest,rgb,color_order='RGB')
            np.testing.assert_array_equal(cv2.imread(str(dest))[...,::-1],result.render_debug(rgb,color_order='RGB'))
            with self.assertRaises(FileExistsError):result.save_debug(dest,rgb,color_order='RGB')
            result.save_debug(dest,rgb[...,::-1],color_order='BGR',overwrite=True)
            np.testing.assert_array_equal(cv2.imread(str(dest))[...,::-1],result.render_debug(rgb,color_order='RGB'))

    def test_symlink_target_not_touched(self):
        result,z=next(self.results())
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);target=p/'target';target.write_bytes(b'keep');link=p/'view.png';link.symlink_to(target)
            with self.assertRaises(FileExistsError):result.save_debug(link,depth=z)
            result.save_debug(link,depth=z,overwrite=True)
            self.assertFalse(link.is_symlink())
            self.assertEqual(target.read_bytes(),b'keep')
            directory=p/'alias';directory.symlink_to(p,target_is_directory=True)
            with self.assertRaises(ValueError):result.save_debug(directory/'new.png',depth=z)

    def test_shape_order_and_options_validation(self):
        result,z=next(self.results())
        for kw in ({'color_order':'auto'},{'depth_scale_m':0},{'mask_alpha':2},{'max_lines':0},{'panel':1}):
            with self.assertRaises(ValueError):result.render_debug(depth=z,**kw)
        with self.assertRaises(ValueError):result.render_debug(np.zeros((3,4,3),np.uint8))
        with self.assertRaises(ValueError):result.render_debug(np.zeros((*z.shape,3)))

    def test_diagnostic_masks_do_not_change_lift_or_plane_measurements(self):
        z=np.full((40,60),2.);m=np.ones(z.shape,bool);k=CameraIntrinsics(100,100,30,20);t=TablePlane((0,0,-1),2)
        d=LiftDetector();self.assertEqual(d.detect(z,k,t,m).to_dict(),d.detect(z,k,t,m,diagnostics=True).to_dict())
        self.assertEqual(fit_table_plane(z,k,m,depth_scale_m=1).to_dict(),
            fit_table_plane(z,k,m,depth_scale_m=1,diagnostics=True).to_dict())

    def test_corner_depth_only_preserves_color_api(self):
        from depth_image_lens import OverheadCornerDetector
        z=np.full((80,100),1000,np.uint16);m=np.zeros(z.shape,bool);m[15:65,20:80]=True
        color=np.full((*z.shape,3),128,np.uint8)
        r=OverheadCornerDetector().detect(color,z,color_order='RGB',registered_to_depth=True,object_mask=m)
        np.testing.assert_array_equal(r.render_debug(depth=z),r.render_debug(color,color_order='RGB'))
        np.testing.assert_array_equal(r.render_debug(depth=z,color_order='BGR'),r.render_debug(depth=z)[...,::-1])
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'corner.png';r.save_debug(p,depth=z)
            np.testing.assert_array_equal(cv2.imread(str(p))[...,::-1],r.render_debug(depth=z))
        with self.assertRaises(ValueError):r.render_debug()
