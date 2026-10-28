"""Optional empty-background contracts, not real reference accuracy claims."""
import json,unittest
import numpy as np
from depth_image_lens import OverheadBoundaryDetector


def scene():
    y,x=np.indices((100,140));background=np.empty((100,140,3),np.uint8)
    background[:]=np.stack([70+x%17,85+y%13,90+(x+y)%11],axis=-1)
    mask=np.zeros((100,140),bool);mask[25:75,35:105]=True
    color=background.copy();color[mask]=(170,150,70)
    check=np.zeros(mask.shape,bool);check[:15,15:125]=True
    depth=np.full(mask.shape,1400,np.uint16)
    return color,background,mask,check,depth


class BackgroundBoundaryTests(unittest.TestCase):
    def call(self,color,bg,**kw):
        return OverheadBoundaryDetector().detect(color,color_order='RGB',background_color=bg,**kw)

    def test_legacy_background_difference(self):
        color,bg,m,c,z=scene();r=self.call(color,bg)
        np.testing.assert_array_equal(r.object_mask,m)
        self.assertEqual(r.stages['background_reference']['status'],'unchecked_candidate')

    def test_illumination_compensation_uses_declared_table(self):
        color,bg,m,c,z=scene();color=np.clip(color.astype(int)+20,0,255).astype(np.uint8)
        r=OverheadBoundaryDetector(background_compensate=True).detect(color,color_order='RGB',background_color=bg,background_check_mask=c)
        self.assertEqual(r.stages['background_reference']['status'],'checked_candidate')
        self.assertGreater(np.linalg.norm(r.stages['background_reference']['encoded_lab_offset']),10)
        self.assertGreater((r.object_mask&m).sum()/m.sum(),.99)
        self.assertLess((r.object_mask&~m).sum(),10)

    def test_no_automatic_illumination_estimate_from_object(self):
        color,bg,m,c,z=scene()
        r=OverheadBoundaryDetector(background_compensate=True).detect(color,color_order='RGB',background_color=bg)
        self.assertEqual(r.stages['background_reference']['status'],'insufficient')
        self.assertEqual(r.stages['segmentation']['mode'],'border_table_assumption')
        self.assertTrue(r.uncertain_mask.all())

    def test_mismatched_reference_rejected_without_vetoing_depth(self):
        color,bg,m,c,z=scene();bg=np.full_like(bg,(200,10,10))
        r=OverheadBoundaryDetector().detect(color,z,color_order='RGB',registered_to_depth=True,background_color=bg,background_check_mask=c)
        self.assertEqual(r.stages['background_reference']['status'],'rejected')
        self.assertGreater(r.stages['background_reference']['check_corrected_p95'],20)
        self.assertEqual(r.stages['depth']['status'],'computed')
        self.assertTrue(r.uncertain_mask.all())

    def test_bad_optional_image_or_mask_falls_back(self):
        color,bg,m,c,z=scene()
        for ref,extra in [(bg[:-1],{}),(bg.astype(float),{}),(bg,{'background_valid_mask':m.astype(np.uint8)}),(bg,{'background_shift_xy':(.5,0)})]:
            r=self.call(color,ref,**extra)
            self.assertEqual(r.stages['background_reference']['status'],'insufficient')
            self.assertTrue(r.object_mask.any());json.dumps(r.to_dict(),allow_nan=False)

    def test_explicit_mask_overrides_bad_reference_including_empty(self):
        color,bg,m,c,z=scene()
        for chosen in [m,np.zeros_like(m)]:
            r=self.call(color,bg[:-1],object_mask=chosen)
            np.testing.assert_array_equal(r.object_mask,chosen)
            self.assertEqual(r.stages['background_reference']['status'],'ignored')

    def test_reference_missingness_not_promoted(self):
        color,bg,m,c,z=scene();valid=np.ones_like(m);valid[20:80,30:110]=False
        r=self.call(color,bg,background_valid_mask=valid,diagnostics=True)
        self.assertFalse(r.object_mask[~valid].any());self.assertTrue(r.uncertain_mask[~valid].all())
        self.assertTrue(np.isnan(r.diagnostics['background_distance'][~valid]).all())

    def test_translation_does_not_fill_reference_border(self):
        color,bg,m,c,z=scene();cv=__import__('cv2')
        shifted=cv.warpAffine(bg,np.array([[1.,0.,-2],[0.,1.,1]]),(140,100))
        valid=np.ones_like(m);valid[:1]=False;valid[:,-2:]=False
        r=self.call(color,shifted,background_shift_xy=(2,-1),background_valid_mask=valid,diagnostics=True)
        self.assertFalse(r.object_mask[:,:2].any());self.assertTrue(r.uncertain_mask[:,:2].all())
        self.assertGreater((r.object_mask&m).sum()/m.sum(),.99)

    def test_contamination_outside_check_cannot_be_certified(self):
        color,bg,m,c,z=scene();contaminated=bg.copy();contaminated[m]=color[m]
        r=self.call(color,contaminated,background_check_mask=c)
        self.assertFalse(r.object_mask.any())
        self.assertIn('check_region_does_not_validate_covered_or_contaminated_reference_regions',r.stages['background_reference']['warnings'])
        self.assertEqual(r.layer_status,'unknown')

    def test_bgr_noncontiguous_and_nonmutation(self):
        color,bg,m,c,z=scene();a=color.copy();b=bg.copy();d=OverheadBoundaryDetector()
        r=d.detect(color,color_order='RGB',background_color=bg,diagnostics=True)
        t=d.detect(color[...,::-1],color_order='BGR',background_color=bg[...,::-1],diagnostics=True)
        np.testing.assert_array_equal(r.object_mask,t.object_mask)
        np.testing.assert_array_equal(r.render_debug(color),t.render_debug(color[...,::-1],color_order='BGR')[...,::-1])
        np.testing.assert_array_equal(color,a);np.testing.assert_array_equal(bg,b)

    def test_all_invalid_reference_and_empty_check(self):
        color,bg,m,c,z=scene()
        r=self.call(color,bg,background_valid_mask=np.zeros_like(m))
        self.assertFalse(r.object_mask.any());self.assertTrue(r.uncertain_mask.all())
        self.assertEqual(r.stages['background_reference']['reason'],'no_valid_background_pixels')
        r=self.call(color,bg,background_valid_mask=np.zeros_like(m),background_check_mask=c)
        self.assertFalse(r.object_mask.any());self.assertTrue(r.uncertain_mask.all())
        self.assertEqual(r.stages['background_reference']['reason'],'no_valid_background_pixels')
        r=self.call(color,bg,background_check_mask=np.zeros_like(m))
        self.assertEqual(r.stages['background_reference']['status'],'insufficient')

    def test_background_parameter_contracts(self):
        for kw in [{'background_compensate':1},{'background_check_p95':0},{'background_max_offset':float('nan')}]:
            with self.assertRaises(ValueError):OverheadBoundaryDetector(**kw)
        color,bg,m,c,z=scene();r=OverheadBoundaryDetector().detect(color,color_order='RGB',background_color=None)
        self.assertNotIn('background_reference',r.stages)


if __name__=='__main__':unittest.main()
