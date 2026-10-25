"""Corner rendering contracts; visualization is not detection evidence."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from depth_image_lens import CornerResult, OverheadCornerDetector
from .corner_cases import cases


@unittest.skipUnless(importlib.util.find_spec("cv2"), "optional OpenCV missing")
class CornerDebugTests(unittest.TestCase):
    def result(self, diagnostics=True):
        c=next(cases())
        return c,OverheadCornerDetector().detect(c['color'],c['depth'],color_order='RGB',
            registered_to_depth=True,object_mask=c['mask'],diagnostics=diagnostics)

    def test_render_order_noncontiguous_and_nonmutation(self):
        c,r=self.result();before=copy.deepcopy(r);color=c['color'].copy()
        rgb=r.render_debug(color,color_order='RGB')
        bgr=r.render_debug(color[...,::-1],color_order='BGR')
        np.testing.assert_array_equal(rgb,bgr[...,::-1])
        np.testing.assert_array_equal(color,c['color']);self.assertEqual(r.to_dict(),before.to_dict())
        np.testing.assert_array_equal(r.object_mask,before.object_mask)
        for key in ('interior_mask','valid_depth_mask','component_labels'):
            np.testing.assert_array_equal(r.diagnostics[key],before.diagnostics[key])
        for x,y in zip(r.diagnostics['contours_xy'],before.diagnostics['contours_xy']):
            np.testing.assert_array_equal(x,y)
        self.assertEqual(rgb.dtype,np.uint8);self.assertGreater(rgb.shape[1],color.shape[1])

    def test_source_only_dimensions_and_clutter_options(self):
        c,r=self.result();a=r.render_debug(c['color'],color_order='RGB',panel=False,max_labels=0,show_contours=False,mask_alpha=0)
        self.assertEqual(a.shape,c['color'].shape)
        np.testing.assert_array_equal(a[0,0],c['color'][0,0])

    def test_status_colors_are_rgb_and_different(self):
        mask=np.zeros((60,150),bool);color=np.zeros((60,150,3),np.uint8)
        candidates=[dict(xy=[20+50*i,30],status=s,reasons=[],evidence={})
            for i,s in enumerate(('accepted','suspected','unknown'))]
        r=CornerResult(candidates,{},mask)
        a=r.render_debug(color,color_order='RGB',panel=False,max_labels=0,mask_alpha=0)
        for i,target in enumerate(((30,210,90),(245,90,70),(255,195,40))):
            tile=a[23:38,13+50*i:28+50*i]
            self.assertTrue(np.any(np.all(tile==target,axis=2)))

    def test_empty_and_missing_diagnostics_render_without_invention(self):
        color=np.full((12,16,3),40,np.uint8);r=CornerResult([],{},np.zeros((12,16),bool))
        plain=r.render_debug(color,color_order='RGB',panel=False,mask_alpha=0)
        np.testing.assert_array_equal(plain,color)
        image=r.render_debug(color,color_order='RGB');self.assertGreater(image.shape[1],16)
        self.assertEqual(r.diagnostics,{})
        c,r=self.result(False);self.assertTrue(r.candidates)
        self.assertEqual(r.diagnostics,{})
        r.render_debug(c['color'],color_order='RGB');self.assertEqual(r.diagnostics,{})

    def test_partial_candidate_fields_and_off_image_positions(self):
        r=CornerResult([{'xy':[2,2],'status':'unknown'}, {'xy':[-5,2],'status':'unknown'},
            {'status':'unknown'}],{},np.zeros((8,8),bool))
        r.render_debug(np.zeros((8,8,3),np.uint8),color_order='RGB')
        self.assertNotIn('evidence',r.candidates[0])

    def test_validation(self):
        c,r=self.result()
        for kw in ({'color_order':'auto'},{'mask_alpha':-1},{'mask_alpha':float('nan')},
                   {'max_labels':-1},{'max_labels':True},{'show_contours':1},{'panel':None}):
            args=dict(color_order='RGB');args.update(kw)
            with self.assertRaises(ValueError):r.render_debug(c['color'],**args)
        for color in (c['color'].astype(float),c['color'][:2],c['color'][...,0]):
            with self.assertRaises(ValueError):r.render_debug(color,color_order='RGB')

    def test_crop_result_uses_crop_coordinates(self):
        c=next(cases());color=c['color'][30:,35:];depth=c['depth'][30:,35:];mask=c['mask'][30:,35:]
        r=OverheadCornerDetector().detect(color,depth,color_order='RGB',registered_to_depth=True,object_mask=mask)
        a=r.render_debug(color,color_order='RGB',panel=False)
        self.assertEqual(a.shape,color.shape)

    def test_png_roundtrip_both_orders_and_explicit_overwrite(self):
        import cv2
        c,r=self.result()
        with tempfile.TemporaryDirectory() as td:
            for order,color in [('RGB',c['color']),('BGR',c['color'][...,::-1])]:
                path=Path(td)/(order+'.png');expected=r.render_debug(color,color_order=order)
                self.assertEqual(r.save_debug(path,color,color_order=order),path)
                decoded=cv2.imread(str(path));decoded=decoded[...,::-1] if order=='RGB' else decoded
                np.testing.assert_array_equal(decoded,expected)
                previous=path.read_bytes()
                with self.assertRaises(FileExistsError):r.save_debug(path,color,color_order=order)
                self.assertEqual(path.read_bytes(),previous)
                r.save_debug(path,color,color_order=order,overwrite=True,panel=False)
                self.assertEqual(cv2.imread(str(path)).shape,color.shape)
            self.assertFalse(list(Path(td).glob('.depth-debug-*')))

    def test_save_symlink_policy_and_invalid_destination(self):
        c,r=self.result()
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);target=base/'target';target.write_bytes(b'preserve')
            path=base/'image.png';path.symlink_to(target)
            with self.assertRaises(FileExistsError):r.save_debug(path,c['color'],color_order='RGB')
            r.save_debug(path,c['color'],color_order='RGB',overwrite=True)
            self.assertEqual(target.read_bytes(),b'preserve');self.assertFalse(path.is_symlink())
            directory=base/'link';directory.symlink_to(base,target_is_directory=True)
            for dest in (directory/'x.png',base/'missing'/'x.png',base/'x.jpg'):
                with self.assertRaises(ValueError):r.save_debug(dest,c['color'],color_order='RGB')
            self.assertFalse((base/'missing').exists())
