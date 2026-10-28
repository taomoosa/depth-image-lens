"""Uniform-color shadow contracts; these analytic checks are not field accuracy."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from depth_image_lens import OverheadBoundaryDetector


def scene(object_color=(185,80,60),table_color=(110,160,190)):
    b=np.full((100,140,3),table_color,np.uint8);c=b.copy()
    obj=np.zeros((100,140),bool);obj[25:80,60:120]=1;c[obj]=object_color
    shadow=np.zeros_like(obj);shadow[40:90,10:50]=1;c[shadow]=np.rint(c[shadow]*.5).astype(np.uint8)
    seed=np.zeros_like(obj);seed[30:40,70:80]=1;check=np.zeros_like(obj);check[5:15,10:30]=1
    return c,b,obj,shadow,seed,check


def detect(c,b,seed,check,**params):
    return OverheadBoundaryDetector(background_shadow_mode='uniform_color',**params).detect(
        c,color_order='RGB',background_color=b,background_check_mask=check,object_color_seed=seed,diagnostics=True)


class ShadowTests(unittest.TestCase):
    def test_suppresses_table_shadow_keeps_chromatic_object(self):
        c,b,m,s,seed,check=scene();r=detect(c,b,seed,check)
        self.assertTrue(r.object_mask[m].all());self.assertFalse(r.object_mask[s].any())
        self.assertTrue(r.diagnostics['shadow_suppressed_mask'][s].all())
        self.assertTrue(r.uncertain_mask[s].all());self.assertEqual(r.stages['shadow_separation']['status'],'computed')
        json.dumps(r.to_dict(),allow_nan=False)

    def test_noise_and_object_shadow(self):
        c,b,m,s,seed,check=scene();c[50:75,65:115]=np.rint(c[50:75,65:115]*.55).astype(np.uint8)
        c=np.clip(c.astype(int)+np.random.default_rng(13).integers(-2,3,c.shape),0,255).astype(np.uint8)
        r=detect(c,b,seed,check);self.assertGreater(r.object_mask[m].mean(),.99);self.assertLess(r.object_mask[s].mean(),.01)

    def test_no_seed_does_not_discard_dark_changes(self):
        c,b,m,s,seed,check=scene();r=detect(c,b,None,check)
        self.assertTrue(r.object_mask[s].all());self.assertTrue(r.uncertain_mask[s].all())
        self.assertFalse(r.diagnostics['shadow_suppressed_mask'].any());self.assertEqual(r.stages['shadow_separation']['status'],'insufficient')

    def test_brightness_only_and_near_color_are_ambiguous(self):
        for object_color,table in [((90,90,90),(180,180,180)),((55,80,95),(110,160,190)),((100,150,180),(110,160,190))]:
            c,b,m,s,seed,check=scene(object_color,table);r=detect(c,b,seed,check)
            self.assertFalse(r.diagnostics['shadow_suppressed_mask'].any())
            self.assertEqual(r.stages['shadow_separation']['status'],'insufficient')
            self.assertTrue(r.uncertain_mask[r.object_mask].all())

    def test_clipped_and_dark_pixels_not_suppressed(self):
        c,b,m,s,seed,check=scene();c[45:55,15:25]=0;c[60:70,15:25]=255
        r=detect(c,b,seed,check)
        self.assertTrue(r.object_mask[45:55,15:25].all());self.assertTrue(r.object_mask[60:70,15:25].all())
        self.assertTrue(r.diagnostics['shadow_unreliable_mask'][45:55,15:25].all())
        self.assertFalse(r.diagnostics['shadow_suppressed_mask'][45:55,15:25].any())

    def test_nonuniform_reference_and_seed_keep_evidence(self):
        c,b,m,s,seed,check=scene();b[50:]=(190,80,60)
        r=detect(c,b,seed,check);self.assertFalse(r.diagnostics['shadow_suppressed_mask'].any())
        self.assertIn('reference_not_approximately_uniform_chromaticity',r.stages['shadow_separation']['reasons'])
        c,b,m,s,seed,check=scene();c[30:35,70:80]=(40,190,100)
        r=detect(c,b,seed,check);self.assertFalse(r.diagnostics['shadow_suppressed_mask'].any())

    def test_invalid_optional_seeds_and_check_continue(self):
        c,b,m,s,seed,check=scene()
        for bad in [seed[:-1],seed.astype(np.uint8),np.zeros_like(seed)]:
            r=detect(c,b,bad,check);self.assertEqual(r.stages['shadow_separation']['status'],'insufficient');self.assertTrue(r.object_mask[m].all())
        r=detect(c,b,seed,None);self.assertEqual(r.stages['shadow_separation']['status'],'insufficient')
        overlap=check|seed;r=detect(c,b,seed,overlap);self.assertEqual(r.stages['shadow_separation']['status'],'insufficient')

    def test_missing_depth_and_missing_reference(self):
        c,b,m,s,seed,check=scene();d=OverheadBoundaryDetector(background_shadow_mode='uniform_color')
        r=d.detect(c,np.zeros((2,2)),color_order='RGB',background_color=b,background_check_mask=check,object_color_seed=seed)
        self.assertTrue(r.object_mask[m].all());self.assertEqual(r.stages['depth']['status'],'insufficient')
        q=d.detect(c,color_order='RGB',object_color_seed=seed)
        self.assertEqual(q.stages['shadow_separation']['status'],'insufficient')

    def test_explicit_mask_priority(self):
        c,b,m,s,seed,check=scene();d=OverheadBoundaryDetector(background_shadow_mode='uniform_color')
        r=d.detect(c,color_order='RGB',background_color=b[:-1],object_mask=m,object_color_seed=seed[:-1])
        np.testing.assert_array_equal(r.object_mask,m);self.assertEqual(r.stages['shadow_separation']['status'],'ignored')

    def test_validity_and_shift(self):
        c,b,m,s,seed,check=scene();valid=np.ones_like(m);valid[50:60,75:85]=0
        r=OverheadBoundaryDetector(background_shadow_mode='uniform_color').detect(c,color_order='RGB',background_color=b,background_valid_mask=valid,background_shift_xy=(1,0),background_check_mask=check,object_color_seed=seed,diagnostics=True)
        self.assertFalse(r.object_mask[:,0].any());self.assertTrue(r.uncertain_mask[:,0].all())
        self.assertFalse(r.object_mask[50:60,76:86].any())

    def test_global_offset_then_shadow_and_rejected_reference(self):
        c,b,m,s,seed,check=scene();c=np.clip(c.astype(int)+10,0,255).astype(np.uint8)
        r=detect(c,b,seed,check,background_compensate=True)
        self.assertEqual(r.stages['background_reference']['status'],'checked_candidate')
        self.assertGreater(r.object_mask[m].mean(),.99)
        bad=np.full_like(b,(240,20,30));q=detect(c,bad,seed,check)
        self.assertEqual(q.stages['background_reference']['status'],'rejected');self.assertEqual(q.stages['shadow_separation']['status'],'insufficient')

    def test_parameter_extremes_and_roundtrip(self):
        for kw in [dict(background_shadow_mode='unknown'),dict(shadow_chroma_threshold=0),dict(shadow_chroma_threshold=np.inf),dict(shadow_intensity_floor=0),dict(shadow_min_ratio=0),dict(shadow_max_ratio=.5)]:
            with self.assertRaises(ValueError):OverheadBoundaryDetector(**kw)
        c,b,m,s,seed,check=scene();r=detect(c,b,seed,check,shadow_min_ratio=.8)
        self.assertTrue(r.object_mask[s].all())
        d=OverheadBoundaryDetector(background_shadow_mode='uniform_color');self.assertEqual(d.parameters,OverheadBoundaryDetector.from_config(d.parameters).parameters)

    def test_RGB_BGR_nonmutation_diagnostics_save(self):
        c,b,m,s,seed,check=scene();inputs=[a.copy() for a in (c,b,seed,check)]
        a=detect(c,b,seed,check);d=OverheadBoundaryDetector(background_shadow_mode='uniform_color')
        q=d.detect(c[...,::-1],color_order='BGR',background_color=b[...,::-1],background_check_mask=check,object_color_seed=seed,diagnostics=True)
        np.testing.assert_array_equal(a.object_mask,q.object_mask)
        np.testing.assert_array_equal(a.render_debug(c),q.render_debug(c[...,::-1],color_order='BGR')[...,::-1])
        off=d.detect(c,color_order='RGB',background_color=b,background_check_mask=check,object_color_seed=seed)
        self.assertEqual(a.to_dict(),off.to_dict());self.assertFalse(off.diagnostics)
        for old,new in zip(inputs,(c,b,seed,check)):np.testing.assert_array_equal(old,new)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'shadow.png';a.save_debug(p,c)
            with self.assertRaises(FileExistsError):a.save_debug(p,c)


if __name__=='__main__':unittest.main()
