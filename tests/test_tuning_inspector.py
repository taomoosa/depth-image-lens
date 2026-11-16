"""Raw diagnostic inspection never recomputes segmentation."""
import unittest
import numpy as np
from depth_image_lens.tuning_backend import PipelineResult, inspect_pixel, diagnostic_metadata, array_image

class InspectorTests(unittest.TestCase):
    def result(self):
        arrays={'segmenter/mean':np.array([[.1,.2],[np.nan,.4]]),
                'segmenter/reason_depth':np.array([[True,False],[False,True]]),
                'segmenter/reason_color':np.array([[True,True],[False,False]]),
                'segmenter/model_all_color_distances':np.arange(8.).reshape(2,2,2),
                'segmenter/model_diagnostic_ids':np.array([4,9])}
        return PipelineResult({},arrays,{'transform':{'crop':[10,20,2,2]},'diagnostics':{'segmenter':{
            'diagnostic_metadata':{'maps':{'mean':{'unit':'m','threshold':.01}}}}}},np.ones((2,2),bool),1.)

    def test_numeric_metadata_and_overlapping_reasons(self):
        info=inspect_pixel(self.result(),'segmenter/mean',0,0)
        self.assertEqual(info['value'],.1)
        self.assertEqual(info['metadata']['unit'],'m')
        self.assertEqual(info['normalized_xy'],[10,20])
        self.assertEqual(info['active_reasons'],['reason_depth','reason_color'])

    def test_model_id_and_nonfinite(self):
        self.assertEqual(inspect_pixel(self.result(),'segmenter/model_all_color_distances',1,0,1)['model_id'],9)
        self.assertEqual(inspect_pixel(self.result(),'segmenter/model_all_color_distances',1,0,1)['value'],5.)
        self.assertTrue(np.isnan(inspect_pixel(self.result(),'segmenter/mean',0,1)['value']))

    def test_composite_and_outside_rejected(self):
        for key,x,y in [('debug',0,0),('segmenter/mean',3,0)]:
            with self.assertRaises(ValueError):inspect_pixel(self.result(),key,x,y)

    def test_categorical_palette_and_decision_legend(self):
        from depth_image_lens.tuning_backend import diagnostic_image
        pixels=diagnostic_image('segmenter/region_label_map',np.array([[0,1,2,193]]))
        self.assertEqual(len(np.unique(pixels.reshape(-1,3),axis=0)),4)
        result=self.result(); result.summary['stages']={'segmenter':{'stages':{'fusion':{'decision_reasons':{'3':'insufficient support'}}}}}
        result.arrays['segmenter/fusion_decision']=np.full((2,2),3)
        self.assertEqual(inspect_pixel(result,'segmenter/mean',0,0)['fusion_decision']['meaning'],'insufficient support')
        self.assertEqual(diagnostic_metadata(result,'segmenter/reason_decision_3_mask')['decision_meaning'],'insufficient support')

    def test_constant_nonzero_distinct_from_zero(self):
        self.assertTrue((array_image(np.ones((2,2)))==160).all())
        self.assertTrue((array_image(np.zeros((2,2)))==0).all())
        np.testing.assert_array_equal(array_image(np.full((1,1),np.nan))[0,0],[255,0,255])

try:
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6 import QtWidgets, QtCore
    from depth_image_lens.tuner import TunerWindow, LatestRunner
    QT_AVAILABLE=True
except ImportError:
    QT_AVAILABLE=False

@unittest.skipUnless(QT_AVAILABLE,'optional GUI dependency absent')
class CanvasInspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        class Executor:
            calls=[]
            def submit(self,*args):raise AssertionError('Inspection must not schedule work')
            def shutdown(self,**kwargs):pass
        self.runner=LatestRunner(executor=Executor());self.runner.timer.stop()
        self.window=TunerWindow(self.runner);self.window.auto.setChecked(False)
        result=InspectorTests().result()
        result.arrays['color']=np.zeros((2,2,3),np.uint8)
        result.images={'segmenter/mean':array_image(result.arrays['segmenter/mean']),
                       'segmenter/reason_depth':array_image(result.arrays['segmenter/reason_depth'])}
        self.window._submitted_project=self.window.project()
        self.window.show_result(result)

    def tearDown(self):
        self.window.close()
        QtCore.QCoreApplication.sendPostedEvents(None,QtCore.QEvent.Type.DeferredDelete)

    def test_letterbox_click_and_overlay(self):
        self.window.fit.setChecked(False)
        self.window.canvas.resize(100,80)
        self.window.view.setCurrentText('segmenter/mean')
        self.window.inspect_canvas(49.5,39.5)
        self.assertIn('"value": 0.1',self.window.inspector.toPlainText())
        self.assertIn('"unit": "m"',self.window.inspector.toPlainText())
        old=self.window.inspector.toPlainText()
        self.window.inspect_canvas(0,0)
        self.assertEqual(self.window.inspector.toPlainText(),old)
        self.window.view.setCurrentText('segmenter/reason_depth')
        self.assertTrue(self.window.overlay.isEnabled())
        self.window.overlay.setChecked(True)
        self.assertIn('Red=true',self.window.view_note.text())
        self.window.inspect_canvas(49.5,39.5)
        self.assertIn('reason_color',self.window.inspector.toPlainText())

    def test_model_raw_stack_click(self):
        self.window.fit.setChecked(False);self.window.canvas.resize(100,80)
        self.window.view.setCurrentText('segmenter/model_all_color_distances')
        self.window.model.setCurrentIndex(1)
        self.window.inspect_canvas(50.5,39.5)
        self.assertIn('"model_id": 9',self.window.inspector.toPlainText())
        self.assertIn('"value": 5.0',self.window.inspector.toPlainText())
