import unittest
import numpy as np
from depth_image_lens import BridgeDrapeDetector
from depth_image_lens.rgbd_diagnostics import guard_masked_negative_claims
from .rgbd_missingness_cases import validation_cases


class NegativeEvidenceTests(unittest.TestCase):
    def result(self,condition='clean'):
        c=next(c for c in validation_cases() if c['condition']==('empty' if condition=='empty' else 'clean'))
        verified=c['difference']==0
        if condition=='path':c['difference'][:,77:80]=0
        if condition=='anchor':c['difference'][24:32,20:36]=0
        if condition=='remote':c['difference'][:8,:15]=0
        r=BridgeDrapeDetector().detect_16uc1(c['depth'],c['difference'],c['hand_rois'],diagnostics=True)
        return c,r,verified

    def test_removed_path_and_anchor_are_unknown_never_invented_yes(self):
        for condition in ('path','anchor'):
            c,r,verified=self.result(condition)
            guarded=guard_masked_negative_claims(r,c['difference'],verified_nonforeground_mask=verified)
            self.assertEqual(r.bridge,'no');self.assertEqual(guarded.bridge,'unknown')
            self.assertEqual(guarded.features['negative_evidence_inferred_pixels'],0)
            self.assertGreater(guarded.features['unverified_zero_difference_search_pixels'],0)
            if condition=='anchor':self.assertEqual(guarded.drape,'unknown')

    def test_background_can_support_no_but_absent_provenance_cannot(self):
        c,r,verified=self.result('empty')
        guarded=guard_masked_negative_claims(r,c['difference'],verified_nonforeground_mask=verified)
        self.assertEqual((guarded.bridge,guarded.drape),('no','neither'))
        self.assertEqual(guard_masked_negative_claims(r,c['difference']).bridge,'unknown')

    def test_positive_and_remote_exclusion_unchanged(self):
        for condition in ('clean','remote'):
            c,r,_=self.result(condition);guarded=guard_masked_negative_claims(r,c['difference'])
            self.assertEqual(guarded.bridge,'yes');self.assertEqual(guarded.drape,r.drape)

    def test_guard_does_not_modify_arrays_or_default_detector(self):
        c,r,_=self.result('path');before=r.to_dict(include_masks=True);delta=c['difference'].copy()
        guarded=guard_masked_negative_claims(r,c['difference'])
        self.assertEqual(r.to_dict(include_masks=True),before)
        np.testing.assert_array_equal(c['difference'],delta)
        np.testing.assert_array_equal(r.foreground_mask,guarded.foreground_mask)
        self.assertEqual(r.bridge,'no');self.assertEqual(guarded.bridge,'unknown')

    def test_incorrect_attestation_cannot_be_inferred_from_arrays(self):
        c,r,_=self.result('path')
        wrong=guard_masked_negative_claims(r,c['difference'],verified_nonforeground_mask=c['difference']==0)
        self.assertEqual(wrong.bridge,'no')
        with self.assertRaises(ValueError):guard_masked_negative_claims(r,c['difference'],verified_nonforeground_mask=np.ones(c['depth'].shape))


if __name__=='__main__':unittest.main()
