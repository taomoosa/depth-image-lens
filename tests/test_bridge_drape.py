import json
import unittest
import numpy as np
from depth_image_lens import BridgeDrapeDetector

ROIS = {"left":(14,4,26,20), "right":(94,4,106,20)}


def inputs(mask):
    background=np.full(mask.shape,2.)
    depth=background.copy();depth[mask]=1.5
    return depth,background-depth


def scene(kind="flat"):
    mask=np.zeros((100,120),bool)
    if kind in ("flat","u","long_hem"):
        for x in range(20,101):
            top=20+(int(round(35*np.sin(np.pi*(x-20)/80))) if kind=="u" else 0)
            mask[top:(80 if kind=="long_hem" else top+7),x]=True
    if kind in ("left","separate"):mask[19:67,17:24]=True
    if kind in ("right","separate"):mask[19:67,97:104]=True
    return mask


class BridgeDrapeTests(unittest.TestCase):
    def detect(self,mask,**kw):return BridgeDrapeDetector().detect(*inputs(mask),ROIS,**kw)

    def test_01_horizontal_bridge_and_json(self):
        r=self.detect(scene());self.assertEqual((r.bridge,r.large_sag,r.drape),("yes","no","neither"))
        json.dumps(r.to_dict(),allow_nan=False);json.dumps(r.to_dict(include_masks=True),allow_nan=False)
        self.assertNotIn('foreground_mask',r.to_dict());self.assertEqual(r.validation_status,'not_validated')

    def test_02_deep_u(self):
        r=self.detect(scene('u'));self.assertEqual((r.bridge,r.large_sag,r.drape),("yes","yes","both"))
        self.assertGreater(r.features['sag_ratio'],.3)

    def test_03_left_only(self):
        r=self.detect(scene('left'));self.assertEqual((r.bridge,r.large_sag,r.drape),("no","unknown","left"))

    def test_04_right_only(self):
        r=self.detect(scene('right'));self.assertEqual((r.bridge,r.drape),("no","right"))

    def test_05_two_separate_drapes(self):
        r=self.detect(scene('separate'));self.assertEqual((r.bridge,r.drape),("no","both"))

    def test_06_arm_only_connection(self):
        m=np.zeros((100,120),bool);m[7:12,20:101]=True;m[7:19,17:24]=True;m[7:19,97:104]=True
        # All foreground is connected above the hands, but there is no deformable-object bridge.
        arms=np.zeros_like(m);arms[:15]=True
        r=self.detect(m,arm_mask=arms);self.assertEqual((r.bridge,r.drape),("no","neither"))
        self.assertGreater(r.foreground_mask.sum(),r.work_mask.sum())

    def test_07_empty_and_negative_difference(self):
        m=np.zeros((100,120),bool);r=self.detect(m)
        self.assertEqual((r.bridge,r.large_sag,r.drape),("no","unknown","neither"))
        z=np.full(m.shape,2.2);diff=np.full(m.shape,-.2)
        r=BridgeDrapeDetector().detect(z,diff,ROIS);self.assertFalse(r.foreground_mask.any())

    def test_08_missing_depth_small_and_large_holes(self):
        z,diff=inputs(scene());z[23,60]=0
        r=BridgeDrapeDetector().detect(z,diff,ROIS)
        self.assertEqual(r.bridge,'yes');self.assertFalse(r.foreground_mask[23,60]);self.assertTrue(r.invalid_mask[23,60])
        z,diff=inputs(scene());z[:,57:64]=0
        r=BridgeDrapeDetector(max_gap_px=2).detect(z,diff,ROIS)
        self.assertEqual((r.bridge,r.large_sag,r.drape),('unknown','unknown','unknown'))
        z,diff=inputs(scene());diff[:]=np.nan
        r=BridgeDrapeDetector().detect(z,diff,ROIS);self.assertEqual(r.bridge,'unknown');self.assertFalse(r.foreground_mask.any())
        # A zero background encoded as difference=-current cannot become foreground.
        z=np.ones((100,120));r=BridgeDrapeDetector().detect(z,-z,ROIS)
        self.assertTrue(r.invalid_mask.all())
        bgvalid=np.zeros_like(z,dtype=bool);r=BridgeDrapeDetector().detect(z,np.ones_like(z),ROIS,background_valid=bgvalid)
        self.assertFalse(r.foreground_mask.any());self.assertEqual(r.drape,'unknown')

    def test_09_hidden_or_out_of_frame_hand(self):
        r=self.detect(scene('left'),hand_visibility=('visible','hidden'));self.assertEqual(r.drape,'unknown')
        boxes={**ROIS,'right':(114,4,128,20)}
        r=BridgeDrapeDetector().detect(*inputs(scene('left')),boxes)
        self.assertEqual((r.bridge,r.drape),('unknown','unknown'))
        # A user mask that deletes anchors must not create a known negative.
        arm=np.zeros((100,120),bool);arm[:25]=True
        self.assertEqual(self.detect(scene('left'),arm_mask=arm).drape,'unknown')

    def test_10_sign_units_and_input_errors(self):
        z,diff=inputs(scene('u'));det=BridgeDrapeDetector()
        original=det.detect(z,diff,ROIS)
        flipped=det.detect(z,-diff,ROIS,difference_sign='current-background')
        metric=det.detect(z*1000,diff*1000,ROIS,depth_unit='mm',difference_unit='mm')
        self.assertEqual(original.to_dict(),flipped.to_dict());self.assertEqual(original.to_dict(),metric.to_dict())
        for kw in [dict(difference_sign='unknown'),dict(depth_unit='m',difference_unit='mm'),dict(depth_unit='unknown'),dict(down=(0,0)),dict(background_valid=np.ones(z.shape))]:
            with self.subTest(kw=list(kw)),self.assertRaises(ValueError):det.detect(z,diff,ROIS,**kw)
        with self.assertRaises(ValueError):det.detect(z,diff[:-1],ROIS)

    def test_11_long_hem_is_not_large_sag(self):
        r=self.detect(scene('long_hem'));self.assertEqual((r.bridge,r.large_sag,r.drape),('yes','no','both'))
        self.assertLess(r.features['sag_ratio'],.05)

    def test_12_sloped_anchors_gap_and_search_clipping(self):
        m=np.zeros((100,120),bool)
        for x in range(20,101):
            top=20+int(round((x-20)/8));m[top:top+7,x]=True
        boxes={**ROIS,'right':(94,14,106,30)}
        r=BridgeDrapeDetector().detect(*inputs(m),boxes);self.assertEqual((r.bridge,r.large_sag),('yes','no'))
        m=scene();m[:,60]=False
        self.assertEqual(BridgeDrapeDetector(max_gap_px=0).detect(*inputs(m),ROIS).bridge,'no')
        z,diff=inputs(m)
        self.assertEqual(BridgeDrapeDetector(max_gap_px=1).detect(z,diff,ROIS).bridge,'no')
        # Known-zero difference pixels still lie on the same depth surface in this recovery case.
        z[20:27,60]=1.5
        r=BridgeDrapeDetector(max_gap_px=1).detect(z,diff,ROIS);self.assertEqual(r.bridge,'yes');self.assertTrue(r.features['gap_used'])
        m=scene();m[:,55:65]=False
        self.assertEqual(BridgeDrapeDetector(max_gap_px=2).detect(*inputs(m),ROIS).bridge,'no')
        m=scene('left');m[65:,17:24]=True
        r=self.detect(m);self.assertEqual((r.bridge,r.drape),('unknown','unknown'))
        overlap={'left':(10,4,80,20),'right':(60,4,110,20)}
        self.assertEqual(BridgeDrapeDetector().detect(*inputs(scene()),overlap).bridge,'unknown')
        rotated={name:(120-b[2],100-b[3],120-b[0],100-b[1]) for name,b in ROIS.items()}
        down=np.array([0.,-2.]);r=BridgeDrapeDetector().detect(*inputs(np.rot90(scene(),2)),rotated,down=down)
        self.assertEqual((r.bridge,r.large_sag),('yes','no'));self.assertEqual(down.tolist(),[0.,-2.])


if __name__=='__main__':unittest.main(verbosity=2)
