"""Frozen uint16 continuity cases; no claim of real-scene performance."""
import json
import unittest
import numpy as np
from depth_image_lens import BridgeDrapeDetector
from .continuity_cases import cases


class ContinuityTests(unittest.TestCase):
    def result(self,name,mode='auto',**options):
        c=next(c for c in cases() if c['name']==name)
        kw={'hand_rois':c['rois']} if mode=='manual' else {'upper_band':(8,20)}
        return BridgeDrapeDetector(max_gap_px=c['gap']).detect_16uc1(c['depth'],c['difference'],diagnostics=True,**kw,**options)

    def test_typical_and_no_fabricated_paths(self):
        expected={'thin_bridge':'yes','small_gap_disabled':'unknown','small_gap_continuous_enabled':'yes',
                  'small_gap_background_depth':'unknown','small_gap_missing_depth':'unknown',
                  'large_zero_gap':'unknown','adjacent_depth_step':'unknown','independent_drapes':'no',
                  'multiple_first_discontinuous':'yes','diagonal_only_depth_step':'unknown'}
        for name,state in expected.items():
            with self.subTest(name=name):
                r=self.result(name);self.assertEqual(r.bridge,state);json.dumps(r.to_dict(),allow_nan=False)
        self.assertEqual(self.result('independent_drapes').drape,'both')

    def test_inferred_gap_is_distinct_and_depth_checked(self):
        for mode in ('manual','auto'):
            r=self.result('small_gap_continuous_enabled',mode)
            gap=r.diagnostics['inferred_gap_mask']
            self.assertTrue(gap.any());self.assertFalse(np.any(gap&r.foreground_mask))
            self.assertFalse(np.any(gap&r.invalid_mask))
            self.assertFalse(np.any(gap&r.diagnostics['bridge_corridor_mask']))
            self.assertEqual(r.reasons['bridge'],'work_path_with_explicit_small_gap_inference')
            r=self.result('small_gap_background_depth',mode)
            self.assertNotEqual(r.bridge,'yes')
        self.assertTrue(self.result('adjacent_depth_step','manual').diagnostics['depth_rejected_mask'].any())
        c=cases()[0];z=c['depth'];delta=c['difference']
        delta[24,75]=0;z[24,75]=2000
        # A continuous observed route around a hole needs no depth-unsafe shortcut.
        for mode in ('manual','auto'):
            kw={'hand_rois':c['rois']} if mode=='manual' else {'upper_band':(8,20)}
            r=BridgeDrapeDetector(max_gap_px=1).detect_16uc1(z,delta,**kw)
            self.assertEqual(r.bridge,'yes')


    def test_second_candidate_checked_after_first_rejection(self):
        r=self.result('multiple_first_discontinuous')
        evaluations=r.auto['candidate_evaluations']
        self.assertEqual([c['decision'] for c in evaluations],['rejected_depth_discontinuity','selected'])
        self.assertGreater(evaluations[0]['span_px'],evaluations[1]['span_px'])
        self.assertEqual(r.bridge,'yes')

    def test_depth_threshold_shared_manual_auto(self):
        for mode in ('manual','auto'):
            self.assertNotEqual(self.result('adjacent_depth_step',mode).bridge,'yes')
            self.assertEqual(self.result('adjacent_depth_step',mode,depth_jump_m=.5).bridge,'yes')
            with self.assertRaises(ValueError):self.result('thin_bridge',mode,depth_jump_m=0)

    def test_noncontiguous_and_float_equivalence_no_mutation(self):
        c=cases()[2];z=np.repeat(c['depth'],2,axis=1);d=np.repeat(c['difference'],2,axis=1)
        detector=BridgeDrapeDetector(max_gap_px=1)
        opts={'upper_band':(8,20),'diagnostics':True}
        r=detector.detect_16uc1(z[:,::2],d[:,::2],**opts)
        f=detector.detect_auto(c['depth']*.001,c['difference']*.001,difference_mode='positive_foreground',**opts)
        self.assertEqual(r.to_dict(include_masks=True),f.to_dict(include_masks=True))
        np.testing.assert_array_equal(z,np.repeat(c['depth'],2,axis=1))
        np.testing.assert_array_equal(d,np.repeat(c['difference'],2,axis=1))

    def test_roundoff_at_depth_limit_adjacent_and_gap(self):
        for base in (1400,1500):
            for delta in (99,100,101):
                for gap in (0,1):
                    c=cases()[0]
                    raw=np.full(c['depth'].shape,base,np.uint16)
                    raw[:,76:]=base+delta
                    difference=c['difference'].copy()
                    if gap:difference[:,75]=0
                    detector=BridgeDrapeDetector(max_gap_px=gap)
                    for mode in ('manual','auto'):
                        opts={'hand_rois':c['rois']} if mode=='manual' else {'upper_band':(8,20)}
                        for units in ('16UC1','m','mm'):
                            with self.subTest(base=base,delta=delta,gap=gap,mode=mode,units=units):
                                if units=='16UC1':
                                    result=detector.detect_16uc1(raw,difference,**opts)
                                else:
                                    scale=.001 if units=='m' else 1.
                                    method=detector.detect if mode=='manual' else detector.detect_auto
                                    result=method(raw.astype(float)*scale,difference.astype(float)*scale,
                                                  depth_unit=units,difference_unit=units,
                                                  difference_mode='positive_foreground',**opts)
                                self.assertEqual(result.bridge=='yes',delta<=100)
        # A difference of 1e-10 m is far above roundoff, not absorbed as noise.
        for delta in (.1-1e-10,.1,.1+1e-10):
            for gap in (0,1):
                c=cases()[0];raw=np.full(c['depth'].shape,1.5)
                raw[:,76:]=1.5+delta;difference=c['difference'].astype(float)*.001
                if gap:difference[:,75]=0
                for mode in ('manual','auto'):
                    detector=BridgeDrapeDetector(max_gap_px=gap)
                    method=detector.detect if mode=='manual' else detector.detect_auto
                    opts={'hand_rois':c['rois']} if mode=='manual' else {'upper_band':(8,20)}
                    result=method(raw,difference,difference_mode='positive_foreground',**opts)
                    self.assertEqual(result.bridge=='yes',delta<=.1)
