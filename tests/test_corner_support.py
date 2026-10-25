import importlib.util
import unittest
import numpy as np
from depth_image_lens import OverheadCornerDetector
from .corner_support_cases import cases


@unittest.skipUnless(importlib.util.find_spec('cv2'), 'optional OpenCV required')
class CornerSupportTests(unittest.TestCase):
    def square(self,size,policy):
        mask=np.zeros((80,80),bool);mask[20:20+size,20:20+size]=True
        return OverheadCornerDetector(support_policy=policy).detect(
            np.zeros((*mask.shape,3),np.uint8),np.where(mask,980,1000).astype(np.uint16),
            object_mask=mask,color_order='RGB',registered_to_depth=True,mode='refined')

    def test_bounded_adaptation_recovers_short_straight_edges(self):
        self.assertEqual(self.square(19,'fixed').summary['counts']['unknown'],4)
        result=self.square(19,'adaptive')
        self.assertEqual(result.summary['counts']['accepted'],4)
        for c in result.candidates:
            e=c['evidence'];self.assertTrue(e['adaptive_support_used'])
            self.assertLessEqual(e['short_edge_window_px'],.8*e['edge_window_px'])
            self.assertGreaterEqual(min(e['short_edge_support_counts']),4)
            self.assertTrue(all(a<b for a,b in zip(e['short_edge_support_counts'],e['edge_support_counts'])))

    def test_inadequate_long_window_stays_unknown(self):
        result=self.square(12,'adaptive')
        self.assertEqual(result.summary['counts']['unknown'],4)
        self.assertFalse(any(c['evidence']['adaptive_support_used'] for c in result.candidates))
        with self.assertRaises(ValueError):OverheadCornerDetector(support_policy='unbounded')
        self.assertEqual(OverheadCornerDetector().support_policy,'fixed')


    def test_internal_step_stays_suspected_with_adaptive_support(self):
        c=next(c for c in cases() if c['name']=='step')
        r=OverheadCornerDetector(support_policy='adaptive').detect(c['color'],c['depth'],
            object_mask=c['mask'],color_order='RGB',registered_to_depth=True)
        for xy in c['observable_step_corners']:
            corner=min(r.candidates,key=lambda x:np.linalg.norm(np.array(x['polygon_xy'])-xy))
            self.assertEqual(corner['status'],'suspected')
            self.assertIn('internal_depth_step_or_nonplanarity',corner['reasons'])

    def test_negative_visibility_and_component_conditions_keep_their_gate(self):
        for c in cases():
            if c['name'] not in ('rounded_small','rounded_large','crop','holes','components'):continue
            kwargs=dict(object_mask=c['mask'],color_order='RGB',registered_to_depth=True)
            fixed=OverheadCornerDetector().detect(c['color'],c['depth'],**kwargs)
            adaptive=OverheadCornerDetector(support_policy='adaptive').detect(c['color'],c['depth'],**kwargs)
            self.assertEqual(fixed.summary['counts'],adaptive.summary['counts'],c['name'])
            self.assertEqual(fixed.summary['single_layer_guarantee'],False)


if __name__=='__main__':unittest.main()
