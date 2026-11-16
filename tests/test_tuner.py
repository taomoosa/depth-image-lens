"""Optional Qt workbench regression tests; no desktop or camera required."""
import copy
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from concurrent.futures import Future

try:
    from PySide6 import QtWidgets, QtCore
    from depth_image_lens.tuner import LatestRunner, TunerWindow
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


class FakeExecutor:
    def __init__(self):
        self.calls = []
        self.closed = False

    def submit(self, function, payload):
        future = Future()
        self.calls.append((function, copy.deepcopy(payload), future))
        return future

    def shutdown(self, **kwargs):
        self.closed = True


@unittest.skipUnless(QT_AVAILABLE, 'optional PySide6 GUI extra not installed')
class TestLatestRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.executor = FakeExecutor()
        self.runner = LatestRunner(executor=self.executor, function=lambda x: x)
        self.runner.timer.stop()
        self.ready = []; self.errors = []
        self.runner.ready.connect(self.ready.append)
        self.runner.failed.connect(self.errors.append)

    def tearDown(self):
        self.runner.shutdown()
        self.runner.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)

    def test_latest_only_no_backlog(self):
        self.runner.submit({'value': 1})
        self.runner.submit({'value': 2})
        self.runner.submit({'value': 3})
        self.assertEqual(len(self.executor.calls), 1)
        self.executor.calls[0][2].set_result('old')
        self.runner.poll()
        self.assertEqual(self.ready, [])
        self.assertEqual(len(self.executor.calls), 2)
        self.assertEqual(self.executor.calls[1][1], {'value': 3})
        self.executor.calls[1][2].set_result('latest')
        self.runner.poll()
        self.assertEqual(self.ready, ['latest'])

    def test_stale_error_is_not_displayed(self):
        self.runner.submit({'value': 1})
        self.runner.submit({'value': 2})
        self.executor.calls[0][2].set_exception(ValueError('obsolete error'))
        self.runner.poll()
        self.assertEqual(self.errors, [])
        self.executor.calls[1][2].set_exception(ValueError('current error'))
        self.runner.poll()
        self.assertEqual(len(self.errors), 1)
        self.assertIn('current error', self.errors[0])

    def test_cancel_discards_active_and_pending(self):
        self.runner.submit({'value': 1})
        self.runner.submit({'value': 2})
        self.runner.cancel()
        self.executor.calls[0][2].set_result('old')
        self.runner.poll()
        self.assertEqual(self.ready, [])
        self.assertEqual(len(self.executor.calls), 1)

    def test_submit_snapshots_inputs(self):
        payload = {'values': [1]}
        self.runner.submit({'initial': True})
        self.runner.submit(payload)
        payload['values'].append(2)
        self.executor.calls[0][2].set_result(None)
        self.runner.poll()
        self.assertEqual(self.executor.calls[1][1], {'values': [1]})

    def test_shutdown_stops_new_work(self):
        self.runner.shutdown()
        self.runner.submit({'ignored': True})
        self.assertEqual(self.executor.calls, [])
        self.assertTrue(self.executor.closed)


@unittest.skipUnless(QT_AVAILABLE, 'optional PySide6 GUI extra not installed')
class TestTunerWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.executor = FakeExecutor()
        self.runner = LatestRunner(executor=self.executor, function=lambda x: x)
        self.runner.timer.stop()
        self.window = TunerWindow(runner=self.runner)
        self.window.auto.setChecked(False)

    def tearDown(self):
        self.window.close()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)

    def test_slider_drag_only_previews(self):
        self.window.parameter.setCurrentText('depth_threshold_m')
        before = copy.deepcopy(self.window.config)
        self.window.slider.setSliderDown(True)
        self.window.slider.setSliderPosition(600)
        self.assertEqual(self.window.config, before)
        self.assertEqual(len(self.executor.calls), 0)
        self.window.slider.setSliderDown(False)
        self.assertNotEqual(self.window.config, before)
        self.assertEqual(len(self.executor.calls), 0)

    def test_project_is_snapshot(self):
        self.window.paths = {'color': '/tmp/source.png'}
        project = self.window.project()
        self.window.paths['color'] = '/tmp/changed.png'
        self.window.config['segmenter']['depth_threshold_m'] = .123
        self.assertEqual(project['paths']['color'], '/tmp/source.png')
        self.assertNotEqual(project['config']['segmenter']['depth_threshold_m'], .123)

    def test_edit_invalidates_active_before_debounce(self):
        self.window.paths = {'color': '/tmp/source.png'}
        self.window.run()
        self.window.auto.setChecked(True)
        self.window.changed()
        # An older result is invalid before the debounce timer submits a new job.
        self.executor.calls[0][2].set_result('stale: not a PipelineResult')
        self.runner.poll()
        self.assertIsNone(self.window.result)
        self.window.debounce.stop()

    def test_cancel_stops_debounced_run(self):
        from PySide6.QtTest import QTest
        self.window.paths={'color':'unused.png'}
        self.window.auto.setChecked(True)
        self.window.changed()
        self.assertTrue(self.window.debounce.isActive())
        self.window.cancel_pending()
        QTest.qWait(300)
        self.assertEqual(self.executor.calls,[])

    def test_auto_off_stops_debounced_run(self):
        from PySide6.QtTest import QTest
        self.window.paths={'color':'unused.png'}
        self.window.auto.setChecked(True)
        self.window.changed()
        self.window.auto.setChecked(False)
        QTest.qWait(300)
        self.assertEqual(self.executor.calls,[])

    def test_crop_snapshot_and_config_sync(self):
        self.window.crop[0].setValue(10)
        self.window.crop[2].setValue(600)
        self.window.changed()
        self.assertEqual(self.window.project()['config']['crop'],[10,0,600,400])
        self.window.config['crop']=[20,30,200,100]
        self.window.sync_options()
        self.assertEqual([s.value() for s in self.window.crop],[20,30,200,100])
        self.assertEqual(self.window.project()['crop'],[20,30,200,100])

    def test_model_stacks_are_selectable(self):
        import numpy as np
        from depth_image_lens.tuning_backend import PipelineResult
        arrays={'segmenter/model_diagnostic_ids':np.array([2,9]),
                'segmenter/model_all_color_distances':np.arange(32).reshape(2,4,4),
                'segmenter/model_all_rejection_flags':np.zeros((2,4,4),np.uint16),
                'segmenter/model_no_evaluable_model_mask':np.zeros((4,4),bool),
                'segmenter/object_mask':np.ones((4,4),bool)}
        result=PipelineResult({'result/mask':np.zeros((4,4,3),np.uint8)},arrays,{},np.ones((4,4),bool),1.)
        self.window._submitted_project=self.window.project()
        self.window.show_result(result)
        self.window.view.setCurrentText('segmenter/model_all_color_distances')
        self.assertTrue(self.window.model.isEnabled())
        self.window.model.setCurrentIndex(1)
        self.assertEqual(self.window.model.currentText(),'9')
        self.assertFalse(self.window.canvas.pixmap().isNull())
        self.window.view.setCurrentText('segmenter/model_rejection_panels')
        self.assertFalse(self.window.canvas.pixmap().isNull())
