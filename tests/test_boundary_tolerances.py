"""Independent Lab tolerance contracts; synthetic checks are not field accuracy."""
import json,tempfile,unittest,warnings
from pathlib import Path
import numpy as np
from depth_image_lens import OverheadBoundaryDetector
from depth_image_lens._background_difference import separate_lab_selection
from .test_boundary_shadow import scene


def settings(**kw):
    return OverheadBoundaryDetector(background_difference_mode='separate_tolerances',**kw)


def math_result(delta,**kw):
    p=settings(**kw).parameters
    a=np.asarray(delta,float).reshape(1,-1,3)
    return separate_lab_selection(a,np.ones(a.shape[:2],bool),np.zeros(a.shape[:2],bool),p)


class SeparateToleranceTests(unittest.TestCase):
    def test_equal_tolerance_exact_decisions(self):
        c,b,m,s,seed,check=scene()
        for t in [10.,30.,80.]:
            legacy=OverheadBoundaryDetector(color_distance=t).detect(c,color_order='RGB',background_color=b)
            split=settings(background_chroma_tolerance=t,background_luma_tolerance=t).detect(c,color_order='RGB',background_color=b)
            np.testing.assert_array_equal(legacy.object_mask,split.object_mask)
        a,u,info,maps=math_result([[0,30,0],[0,30+1e-8,0],[0,30-1e-8,0],[18,24,0]])
        np.testing.assert_array_equal(a,[[False,True,False,False]])
        self.assertTrue(u.all());self.assertEqual(maps['background_combined_score'][0,3],1.)

    def test_public_encoded_lab_mixed_ellipse_and_equality(self):
        b=np.full((96,128,3),128,np.uint8);c=b.copy();c[20:76,30:98]=(170,85,110)
        tc=np.hypot(38.,2.)/.8;tl=17./.8
        r=settings(background_chroma_tolerance=tc,background_luma_tolerance=tl).detect(c,color_order='RGB',background_color=b,diagnostics=True)
        self.assertAlmostEqual(r.diagnostics['background_combined_score'][30,40],1.28)
        self.assertTrue(r.object_mask[30,40])
        c[20:76,30:98]=(95,95,95)
        for t,want in [(34.,False),(34.-1e-7,True),(34.+1e-7,False)]:
            q=settings(background_luma_tolerance=t).detect(c,color_order='RGB',background_color=b,diagnostics=True)
            self.assertEqual(bool(q.object_mask[30,40]),want);self.assertTrue(q.uncertain_mask[30,40])

    def test_shift_uncovered_pixels_remain_unknown(self):
        c,b,m,s,seed,check=scene();valid=np.ones_like(m);valid[30:40,70:80]=False
        r=settings(background_chroma_tolerance=1e-200,background_luma_tolerance=1e-200).detect(c,color_order='RGB',background_color=b,background_valid_mask=valid,background_shift_xy=(2,1),diagnostics=True)
        self.assertFalse(r.object_mask[:1].any());self.assertFalse(r.object_mask[:,:2].any())
        self.assertTrue(r.uncertain_mask[31:41,72:82].all());self.assertFalse(r.object_mask[31:41,72:82].any())
        self.assertTrue(np.isnan(r.diagnostics['background_combined_score'][:,:2]).all())

    def test_equal_tolerance_score_agrees_at_float_boundary(self):
        delta=[146.,84.,38.];threshold=float(np.linalg.norm(delta))
        a,u,info,m=math_result([delta],background_chroma_tolerance=threshold,background_luma_tolerance=threshold)
        self.assertFalse(a[0,0]);self.assertEqual(m['background_combined_score'][0,0],1.)
        self.assertTrue(u[0,0])

    def test_equal_default_margin_matches_legacy_unclipped(self):
        c,b,m,s,seed,check=scene()
        a=OverheadBoundaryDetector().detect(c,color_order='RGB',background_color=b)
        q=settings().detect(c,color_order='RGB',background_color=b)
        np.testing.assert_array_equal(a.uncertain_mask,q.uncertain_mask)

    def test_luma_tolerance_tradeoff_preserves_chromatic_difference(self):
        a,_,_,_=math_result([[50,0,0],[0,50,0]],background_luma_tolerance=30)
        b,_,_,_=math_result([[50,0,0],[0,50,0]],background_luma_tolerance=80)
        np.testing.assert_array_equal(a,[[True,True]])
        np.testing.assert_array_equal(b,[[False,True]])
        c,_,_,_=math_result([[50,0,0],[0,50,0]],background_chroma_tolerance=80)
        np.testing.assert_array_equal(c,[[True,False]])

    def test_unitless_radius_margin_not_squared_margin(self):
        a,u,info,m=math_result([[2.4,0,0],[3,0,0],[3.6,0,0],[4.2,0,0]],background_chroma_tolerance=4,background_luma_tolerance=3,background_score_margin=.21)
        np.testing.assert_array_equal(u,[[True,True,True,False]])
        np.testing.assert_array_equal(a,[[False,False,True,True]])
        self.assertAlmostEqual(m['background_combined_score'][0,2],1.44)

    def test_numeric_extremes_finite_scores(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for tc,tl in [(np.nextafter(0.,1.),1e308),(1e308,np.nextafter(0.,1.)),(1e-300,1e-300),(1e308,1e308)]:
                a,u,info,m=math_result([[0,0,0],[255,255,255]],background_chroma_tolerance=tc,background_luma_tolerance=tl)
                self.assertTrue(np.isfinite(m['background_combined_score']).all());self.assertLessEqual(m['background_combined_score'].max(),1e12)
                self.assertFalse(a[0,0]);json.dumps(info,allow_nan=False)

    def test_invalid_settings(self):
        for key in ['background_chroma_tolerance','background_luma_tolerance']:
            for value in [0,-1,np.nan,np.inf,True,None]:
                with self.assertRaises(ValueError):settings(**{key:value})
        for value in [-.1,1.1,np.nan,True]:
            with self.assertRaises(ValueError):settings(background_score_margin=value)
        with self.assertRaises(ValueError):OverheadBoundaryDetector(background_difference_mode='weighted')

    def test_no_seed_required_and_strict_json_config_replay(self):
        c,b,m,s,seed,check=scene();d=settings(background_luma_tolerance=60)
        args=dict(color_order='RGB',background_color=b,background_check_mask=check,diagnostics=True)
        r=d.detect(c,**args);q=OverheadBoundaryDetector.from_config(json.loads(json.dumps(d.parameters))).detect(c,**args)
        self.assertEqual(r.to_dict(),q.to_dict());self.assertNotIn('shadow_separation',r.stages)
        self.assertTrue(r.stages['background_difference']['applied_to_selection']);json.dumps(r.to_dict(),allow_nan=False)
        for key in ['background_chroma_distance','background_luma_distance','background_combined_score']:self.assertEqual(r.diagnostics[key].shape,c.shape[:2])

    def test_compensation_precedes_split_thresholds(self):
        c,b,m,s,seed,check=scene();c=np.clip(c.astype(int)+8,0,255).astype(np.uint8)
        kw=dict(color_order='RGB',background_color=b,background_check_mask=check)
        a=OverheadBoundaryDetector(background_compensate=True).detect(c,**kw)
        r=settings(background_compensate=True).detect(c,**kw)
        np.testing.assert_array_equal(a.object_mask,r.object_mask)
        self.assertEqual(a.stages['background_reference']['encoded_lab_offset'],r.stages['background_reference']['encoded_lab_offset'])

    def test_large_tolerances_do_not_bypass_reference_check(self):
        c,b,m,s,seed,check=scene();b[:]=(200,30,30)
        r=settings(background_chroma_tolerance=1e300,background_luma_tolerance=1e300).detect(c,color_order='RGB',background_color=b,background_check_mask=check)
        self.assertEqual(r.stages['background_reference']['status'],'rejected')
        self.assertFalse(r.stages['background_difference']['applied_to_selection'])
        self.assertTrue(r.uncertain_mask.all())

    def test_explicit_empty_and_invalid_masks(self):
        c,b,m,s,seed,check=scene();empty=np.zeros_like(m)
        r=settings().detect(c,color_order='RGB',background_color=b,object_mask=empty,diagnostics=True)
        self.assertFalse(r.object_mask.any());self.assertFalse(r.stages['background_difference']['applied_to_selection'])
        with self.assertRaises(ValueError):settings().detect(c,color_order='RGB',object_mask=empty[:-1])
        q=settings().detect(c,color_order='RGB',background_color=b,background_valid_mask=m[:-1])
        self.assertTrue(q.uncertain_mask.all())

    def test_missing_optional_inputs_continue_and_all_invalid_unknown(self):
        c,b,m,s,seed,check=scene()
        a=settings().detect(c,color_order='RGB');self.assertFalse(a.stages['background_difference']['applied_to_selection'])
        q=settings().detect(c,np.zeros((2,2)),color_order='RGB',background_color=b,depth_scale_m=0)
        self.assertEqual(q.stages['depth']['status'],'insufficient');self.assertTrue(q.object_mask.any())
        r=settings().detect(c,color_order='RGB',background_color=b,background_valid_mask=np.zeros_like(m),diagnostics=True)
        self.assertFalse(r.object_mask.any());self.assertTrue(r.uncertain_mask.all());self.assertTrue(np.isnan(r.diagnostics['background_combined_score']).all())
        json.dumps(r.to_dict(),allow_nan=False)

    def test_clipping_is_uncertain_without_suppressing_foreground(self):
        c,b,m,s,seed,check=scene();c[30:40,70:80]=255
        r=settings().detect(c,color_order='RGB',background_color=b,diagnostics=True)
        self.assertTrue(r.uncertain_mask[30:40,70:80].all());self.assertTrue(r.object_mask[30:40,70:80].all())
        self.assertTrue(r.diagnostics['background_photometric_clipped_mask'][30:40,70:80].all())

    def test_seeded_shadow_composition_is_explicit(self):
        c,b,m,s,seed,check=scene();d=settings(background_shadow_mode='uniform_color')
        r=d.detect(c,color_order='RGB',background_color=b,background_check_mask=check,object_color_seed=seed,diagnostics=True)
        self.assertTrue(r.stages['background_difference']['applied_to_selection'])
        self.assertEqual(r.stages['shadow_separation']['status'],'computed')
        self.assertFalse(r.object_mask[s].any());self.assertTrue(r.object_mask[m].all())

    def test_BGR_noncontiguous_nonmutation_debug_parity_and_save(self):
        c,b,m,s,seed,check=scene();old=[v.copy() for v in [c,b,check]];d=settings(background_luma_tolerance=60)
        kw=dict(background_check_mask=check,diagnostics=True)
        a=d.detect(c,color_order='RGB',background_color=b,**kw);q=d.detect(c[...,::-1],color_order='BGR',background_color=b[...,::-1],**kw)
        np.testing.assert_array_equal(a.object_mask,q.object_mask);self.assertEqual(a.to_dict(),q.to_dict())
        z=d.detect(c,color_order='RGB',background_color=b,background_check_mask=check);self.assertEqual(a.to_dict(),z.to_dict())
        self.assertEqual(a.render_debug(c).dtype,np.uint8)
        for x,y in zip(old,[c,b,check]):np.testing.assert_array_equal(x,y)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'split.png';a.save_debug(path,c)
            with self.assertRaises(FileExistsError):a.save_debug(path,c)


if __name__=='__main__':unittest.main()
