"""Optional, local-only Qt tuning workbench. Importing the core never imports Qt."""
from __future__ import annotations

import copy
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import tempfile

import numpy as np


def execute_job(payload):
    """Run in a spawned process, away from the GUI event loop and its GIL."""
    from . import tuning_backend as backend
    kind = payload.get('kind', 'preview')
    if kind == 'export':
        import cv2
        output=Path(tempfile.mkdtemp(prefix='tuning-result-',dir=payload['directory']))
        (output/'project.json').write_text(json.dumps(payload['project'],indent=2)+'\n',encoding='utf-8')
        result=payload.get('result')
        if result is not None:
            (output/'summary.json').write_text(json.dumps(result.summary,indent=2,default=str)+'\n',encoding='utf-8')
            np.savez_compressed(output/'arrays.npz',**result.arrays)
            for i,(name,array) in enumerate(result.images.items()):
                if not cv2.imwrite(str(output/f'{i:03d}.png'),array[...,::-1]):raise OSError('Image export failed')
            (output/'images.json').write_text(json.dumps(list(result.images),indent=2)+'\n')
        if payload.get('search_result') is not None:
            (output/'search.json').write_text(json.dumps(payload['search_result'],indent=2,default=str)+'\n')
        return {'kind':'export','directory':str(output)}
    if kind == 'preview':
        arrays = {key: backend.load_array(value, key) for key, value in payload['paths'].items() if value}
        prepared = backend.prepare_inputs(arrays, crop=tuple(payload['crop']))
        return backend.run_pipeline(prepared, payload['config'])
    if kind in ('batch', 'search'):
        samples = backend.load_manifest(payload['manifest'])
        prepared_samples = []
        for sample in samples:
            prepared_samples.append({**sample, 'prepared': backend.prepare_inputs(
                sample['arrays'], crop=tuple(payload['crop']), intrinsics=sample.get('intrinsics'))})
        provenance={k:copy.deepcopy(v) for k,v in payload.items() if k!='_cancel_path'}
        cancel = lambda: Path(payload['_cancel_path']).exists()
        if kind == 'batch':
            return {'kind': kind, 'request':provenance, 'rows': backend.evaluate_dataset(prepared_samples, payload['config'], cancel=cancel)}
        training = [s for s in prepared_samples if s.get('split','train') == 'train']
        validation = [s for s in prepared_samples if s.get('split') == 'validation']
        return {'kind':kind, 'request':provenance, **backend.search_parameters(training, payload['config'], payload['space'],
            method=payload['method'], max_trials=payload['max_trials'], seed=payload['seed'],
            objective=payload['objective'], validation_samples=validation, cancel=cancel)}
    raise ValueError(f'Unknown job kind: {kind}')


def validate_project(data):
    from .tuning_backend import STAGES, default_config
    if not isinstance(data,dict) or data.get('version')!=1:raise ValueError('Unsupported project version')
    crop=data.get('crop')
    if not isinstance(crop,list) or len(crop)!=4 or any(type(v) is not int for v in crop):
        raise ValueError('Crop requires four integers: x,y,width,height')
    x,y,w,h=crop
    if x<0 or y<0 or w<2 or h<2 or x+w>640 or y+h>400:raise ValueError('Crop is outside 640x400')
    if not isinstance(data.get('paths'),dict) or any(not isinstance(v,str) for v in data['paths'].values()):
        raise ValueError('Input paths must be strings')
    config=data.get('config')
    if not isinstance(config,dict) or set(config)-set(default_config()):raise ValueError('Invalid configuration keys')
    if config.get('source') not in ('segmenter','mask') or config.get('target') not in ('mask','count','outline'):
        raise ValueError('Invalid source / target')
    scale=config.get('depth_scale_m')
    if isinstance(scale,bool) or not isinstance(scale,(int,float)) or not np.isfinite(scale) or scale<=0:
        raise ValueError('Depth scale must be positive and finite')
    for stage,constructor in STAGES.items():
        options=config.get(stage,{})
        if not isinstance(options,dict):raise ValueError('Stage parameters must be objects')
        if stage!='repair':constructor(**options)
        else:
            import inspect
            if set(options)-set(inspect.signature(constructor).parameters):raise ValueError('Unknown repair parameter')
    for flag in ('diagnostics','repair_enabled'):
        if type(config.get(flag)) is not bool:raise ValueError(flag+' must be boolean')


try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as exc:
    raise ImportError('The tuner requires the optional GUI extra: pip install "depth-image-lens[gui]"') from exc


class LatestRunner(QtCore.QObject):
    """One running job plus one replaceable latest request; never an event backlog."""
    ready = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    state = QtCore.Signal(str)

    def __init__(self, parent=None, executor=None, function=execute_job):
        super().__init__(parent)
        self.executor = executor or ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'))
        self.function = function
        self.generation = 0
        self.active = None
        self.pending = None
        self.closed = False
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.poll)
        self.timer.start()

    def submit(self, payload):
        if self.closed:
            return
        self.generation += 1
        self._cancel_active()
        self.pending = (self.generation, copy.deepcopy(payload))
        self.state.emit('Running; latest settings queued' if self.active else 'Running')
        self.poll()

    def _cancel_active(self):
        path = getattr(self, '_cancel_path', None)
        if path is not None:
            Path(path).touch()

    def cancel(self):
        self._cancel_active()
        self.generation += 1
        self.pending = None
        self.state.emit('Cancelled; current computation will finish without display' if self.active else 'Idle')

    def poll(self):
        if self.closed:
            return
        if self.active is not None and self.active[1].done():
            generation, future = self.active
            self.active = None
            path = getattr(self, '_cancel_path', None)
            if path is not None:
                Path(path).unlink(missing_ok=True)
                self._cancel_path = None
            if self.pending is None:
                self.state.emit('Idle')
            try:
                value = future.result()
                if generation == self.generation:
                    self.ready.emit(value)
            except Exception as exc:
                if generation == self.generation:
                    self.failed.emit(f'{type(exc).__name__}: {exc}')
        if self.active is None and self.pending is not None:
            generation, payload = self.pending
            self.pending = None
            try:
                if payload.get('kind') in ('batch','search'):
                    import os
                    fd, path = tempfile.mkstemp(prefix='depth-tuner-cancel-')
                    os.close(fd); Path(path).unlink()
                    self._cancel_path = path
                    payload['_cancel_path'] = path
                self.active = (generation, self.executor.submit(self.function, payload))
            except Exception as exc:
                self.failed.emit(f'{type(exc).__name__}: {exc}')
                self.state.emit('Idle')

    def shutdown(self):
        if self.closed:return
        self.cancel()
        self.closed = True
        self.timer.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)


class DiagnosticCanvas(QtWidgets.QLabel):
    """Forward positions in label coordinates; the window maps scaled pixels."""
    clicked = QtCore.Signal(float, float)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(event.position().x(), event.position().y())
        super().mousePressEvent(event)


class TunerWindow(QtWidgets.QMainWindow):
    def __init__(self, runner=None):
        super().__init__()
        from .tuning_backend import default_config
        self.config = default_config()
        self.paths = {}
        self.result = None
        self.result_project = None
        self.search_result = None
        self._image = None
        self._building = False
        self.setWindowTitle('Depth Image Lens · Local Tuner')
        self.resize(1280, 850)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.runner = runner or LatestRunner(self)
        if self.runner.parent() is None:self.runner.setParent(self)
        self.runner.ready.connect(self.show_result)
        self.runner.failed.connect(self.show_error)
        self.runner.state.connect(self.statusBar().showMessage)
        central = QtWidgets.QSplitter()
        self.setCentralWidget(central)
        left = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(left)
        form.addWidget(QtWidgets.QLabel('INPUTS · already registered RGB / depth'))
        hint = QtWidgets.QLabel('640×400 / 1280×800 may be mixed. Same aligned view required; resizing is not registration.')
        hint.setWordWrap(True)
        form.addWidget(hint)
        self.path_labels = {}
        for role in ('color', 'depth', 'background_color', 'background_depth', 'mask'):
            row = QtWidgets.QHBoxLayout()
            button = QtWidgets.QPushButton(role)
            button.clicked.connect(lambda checked=False, r=role: self.choose_file(r))
            label = QtWidgets.QLineEdit()
            label.setPlaceholderText('Optional' if role == 'mask' else 'Choose file')
            label.editingFinished.connect(lambda r=role, w=label: self.set_path(r, w.text()))
            self.path_labels[role] = label
            row.addWidget(button); row.addWidget(label)
            form.addLayout(row)
        crop_row = QtWidgets.QHBoxLayout()
        crop_row.addWidget(QtWidgets.QLabel('Crop x,y,w,h at 640×400'))
        self.crop = []
        for value, maximum in ((0,639),(0,399),(640,640),(400,400)):
            spin = QtWidgets.QSpinBox(); spin.setRange(0,maximum); spin.setValue(value)
            spin.editingFinished.connect(self.changed)
            self.crop.append(spin); crop_row.addWidget(spin)
        form.addLayout(crop_row)
        choices = QtWidgets.QHBoxLayout()
        self.source = QtWidgets.QComboBox(); self.source.addItems(['segmenter','mask'])
        self.target = QtWidgets.QComboBox(); self.target.addItems(['mask','count','outline'])
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(['union','depth_required','evidence','color_seeded','depth_anchored','color_supported','color_region','color_local','color_model'])
        for w in (self.source,self.target,self.mode):
            choices.addWidget(w); w.currentTextChanged.connect(self.options_changed)
        form.addLayout(choices)
        controls = QtWidgets.QHBoxLayout()
        self.repair = QtWidgets.QCheckBox('Repair')
        self.details = QtWidgets.QCheckBox('Full diagnostics (slower)')
        self.details.setToolTip('Explicitly enables detailed arrays, including all per-model diagnostics. May use substantial memory.')
        self.auto = QtWidgets.QCheckBox('Auto-run after edit'); self.auto.setChecked(True)
        self.auto.toggled.connect(self.auto_changed)
        for w in (self.repair,self.details): w.toggled.connect(self.options_changed)
        for w in (self.repair,self.details,self.auto): controls.addWidget(w)
        form.addLayout(controls)
        unit_row = QtWidgets.QHBoxLayout()
        unit_row.addWidget(QtWidgets.QLabel('Depth unit'))
        self.unit = QtWidgets.QComboBox(); self.unit.addItems(['millimetres','metres','custom (JSON)'])
        self.unit.currentTextChanged.connect(self.options_changed); unit_row.addWidget(self.unit)
        form.addLayout(unit_row)
        form.addWidget(QtWidgets.QLabel('PARAMETERS · JSON values; null disables optional values'))
        param_row = QtWidgets.QHBoxLayout()
        self.stage = QtWidgets.QComboBox(); self.stage.addItems(['segmenter','repair','counter','outline'])
        self.parameter = QtWidgets.QComboBox(); self.parameter.setMinimumContentsLength(24)
        param_row.addWidget(self.stage); param_row.addWidget(self.parameter)
        form.addLayout(param_row)
        self.value = QtWidgets.QLineEdit(); self.value.editingFinished.connect(self.commit_value)
        form.addWidget(self.value)
        slider_row = QtWidgets.QHBoxLayout()
        self.minimum = QtWidgets.QDoubleSpinBox(); self.maximum = QtWidgets.QDoubleSpinBox()
        for w in (self.minimum,self.maximum):
            w.setRange(-1e9,1e9); w.setDecimals(6); w.editingFinished.connect(self.sync_slider)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0,1000); self.slider.setTracking(False)
        self.slider.sliderMoved.connect(self.preview_slider)
        self.slider.valueChanged.connect(self.commit_slider)
        slider_row.addWidget(self.minimum); slider_row.addWidget(self.slider); slider_row.addWidget(self.maximum)
        form.addLayout(slider_row)
        self.stage.currentTextChanged.connect(self.fill_parameters)
        self.parameter.currentTextChanged.connect(self.select_parameter)
        buttons = QtWidgets.QHBoxLayout()
        for label, callback in [('Run',self.run),('Cancel',self.cancel_pending),('Reset parameters',self.reset_parameters)]:
            button = QtWidgets.QPushButton(label); button.clicked.connect(callback); buttons.addWidget(button)
        form.addLayout(buttons)
        saves = QtWidgets.QHBoxLayout()
        for label, callback in [('Load project',self.load_project),('Save project',self.save_project),('Export result',self.export_result)]:
            button = QtWidgets.QPushButton(label); button.clicked.connect(callback); saves.addWidget(button)
        form.addLayout(saves)
        batch_buttons = QtWidgets.QHBoxLayout()
        for label, callback in [('Batch / search',self.batch_dialog),('Apply best',self.apply_best),('Edit all settings',self.edit_config)]:
            button=QtWidgets.QPushButton(label); button.clicked.connect(callback); batch_buttons.addWidget(button)
        form.addLayout(batch_buttons)
        self.summary = QtWidgets.QPlainTextEdit(); self.summary.setReadOnly(True)
        form.addWidget(self.summary,1)
        central.addWidget(left)
        right = QtWidgets.QWidget(); rlayout = QtWidgets.QVBoxLayout(right)
        toolbar = QtWidgets.QHBoxLayout()
        self.view = QtWidgets.QComboBox(); self.view.currentTextChanged.connect(self.display_image)
        self.fit = QtWidgets.QCheckBox('Fit'); self.fit.setChecked(True); self.fit.toggled.connect(self.display_image)
        self.model = QtWidgets.QComboBox(); self.model.setToolTip('Model ID for the selected diagnostic stack')
        self.model.currentIndexChanged.connect(self.display_image)
        self.overlay = QtWidgets.QCheckBox('Mask overlay')
        self.overlay.toggled.connect(self.display_image)
        self.view_note = QtWidgets.QLabel('Select a debug view; display switching does not rerun detection')
        self.view_note.setWordWrap(True)
        toolbar.addWidget(self.view,1); toolbar.addWidget(self.model); toolbar.addWidget(self.fit); toolbar.addWidget(self.overlay)
        rlayout.addLayout(toolbar); rlayout.addWidget(self.view_note)
        self.canvas = DiagnosticCanvas('Load inputs, then Run')
        self.canvas.clicked.connect(self.inspect_canvas)
        self.canvas.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.canvas.setStyleSheet('background:#161b24;color:#9facbe;')
        self.scroll = QtWidgets.QScrollArea(); self.scroll.setWidget(self.canvas); self.scroll.setWidgetResizable(True)
        rlayout.addWidget(self.scroll,1)
        self.inspector = QtWidgets.QPlainTextEdit(); self.inspector.setReadOnly(True)
        self.inspector.setMaximumHeight(180)
        self.inspector.setPlaceholderText('Click a pixel to inspect raw values and overlapping reasons. No detection is rerun.')
        rlayout.addWidget(self.inspector); central.addWidget(right)
        central.setSizes([475,805])
        self.debounce = QtCore.QTimer(self); self.debounce.setSingleShot(True); self.debounce.setInterval(250); self.debounce.timeout.connect(self.run)
        self.fill_parameters()
        self.statusBar().showMessage('Local only. No camera or robot control. Crop follows resize.')

    def choose_file(self, role):
        path,_ = QtWidgets.QFileDialog.getOpenFileName(self, f'Open {role}', '', 'Images and arrays (*.png *.jpg *.jpeg *.tif *.tiff *.npy *.npz)')
        if path: self.path_labels[role].setText(path); self.set_path(role,path)

    def set_path(self, role, path):
        self.paths[role] = path
        self.changed()

    def options_changed(self, *args):
        if self._building: return
        self.config.update(source=self.source.currentText(), target=self.target.currentText(), repair_enabled=self.repair.isChecked(), diagnostics=self.details.isChecked(), depth_scale_m=(.001,1.,self.config.get('depth_scale_m',.001))[self.unit.currentIndex()])
        self.config['segmenter']['fusion_mode'] = self.mode.currentText()
        self.changed()

    def cancel_pending(self):
        self.debounce.stop()
        self.runner.cancel()

    def auto_changed(self, enabled):
        if not enabled and hasattr(self,'debounce'):self.debounce.stop()

    def changed(self, *args):
        if not self._building: self.runner.cancel()
        if hasattr(self,'debounce') and self.auto.isChecked() and not self._building:
            self.debounce.start()
        self.config['crop']=[s.value() for s in self.crop]
        self.statusBar().showMessage('Settings changed; previous result is stale')

    def fill_parameters(self, *args):
        from .tuning_backend import parameter_defaults
        self._building = True
        self.parameter.clear(); self.parameter.addItems(sorted(parameter_defaults(self.stage.currentText())))
        self._building = False
        self.select_parameter()

    def current_value(self):
        from .tuning_backend import parameter_defaults
        stage,key = self.stage.currentText(),self.parameter.currentText()
        return self.config.get(stage,{}).get(key,parameter_defaults(stage).get(key))

    def select_parameter(self, *args):
        if self._building: return
        value = self.current_value()
        self.value.setText(json.dumps(value))
        numeric = isinstance(value,(int,float)) and not isinstance(value,bool)
        self.slider.setEnabled(numeric)
        self.minimum.setEnabled(numeric); self.maximum.setEnabled(numeric)
        if numeric:
            self.minimum.setValue(min(0.,value*2)); self.maximum.setValue(max(1. if isinstance(value,int) else .01, value*2,abs(value)*2))
            self.sync_slider()

    def sync_slider(self):
        value = self.current_value(); lo,hi=self.minimum.value(),self.maximum.value()
        if isinstance(value,(int,float)) and not isinstance(value,bool) and hi>lo:
            self.slider.blockSignals(True); self.slider.setValue(round(1000*(value-lo)/(hi-lo))); self.slider.blockSignals(False)

    def slider_value(self, position):
        value = self.minimum.value()+(self.maximum.value()-self.minimum.value())*position/1000.
        return int(round(value)) if type(self.current_value()) is int else value

    def preview_slider(self, position):
        self.value.setText(json.dumps(self.slider_value(position)))

    def commit_slider(self, position):
        self.preview_slider(position)
        self.commit_value()

    def commit_value(self):
        try:
            value = json.loads(self.value.text())
            self.config.setdefault(self.stage.currentText(),{})[self.parameter.currentText()] = value
            if self.stage.currentText()=='segmenter' and self.parameter.currentText()=='fusion_mode':
                self.mode.setCurrentText(str(value))
            self.sync_slider(); self.changed()
        except (ValueError,TypeError) as exc: self.show_error(str(exc))

    def reset_parameters(self):
        self.config[self.stage.currentText()] = {}
        if self.stage.currentText()=='segmenter': self.config['segmenter']['fusion_mode']=self.mode.currentText()
        self.select_parameter(); self.changed()

    def run(self):
        self.debounce.stop()
        if not any(self.paths.values()): return
        self.config['crop']=[s.value() for s in self.crop]
        self._submitted_project = copy.deepcopy(self.project())
        self.runner.submit({'kind':'preview','paths':self.paths,'crop':[s.value() for s in self.crop],'config':self.config})

    def show_result(self, result):
        if isinstance(result,dict):
            if result.get('kind')=='export':
                self.statusBar().showMessage('Exported to '+result['directory']);return
            self.search_result=result
            self.summary.setPlainText(json.dumps(result,indent=2,default=str))
            self.statusBar().showMessage('Batch complete; displayed image remains the previous preview')
            return
        self.result_project=copy.deepcopy(self._submitted_project)
        self.result = result
        old=self.view.currentText(); self.view.blockSignals(True); self.view.clear(); self.view.addItems(result.images)
        stacks=[key for key,array in result.arrays.items() if '/model_all_' in key and array.ndim==3 and len(array)>0]
        self.view.addItems(stacks)
        if 'segmenter/model_all_rejection_flags' in stacks:self.view.addItem('segmenter/model_rejection_panels')
        self.model.blockSignals(True);self.model.clear()
        for model_id in result.arrays.get('segmenter/model_diagnostic_ids',[]):self.model.addItem(str(int(model_id)))
        self.model.blockSignals(False)
        if self.view.findText(old)>=0:self.view.setCurrentText(old)
        self.view.blockSignals(False)
        self.summary.setPlainText(json.dumps(result.summary,indent=2,default=str))
        self.display_image()
        self.statusBar().showMessage(f'Completed in {result.elapsed_ms:.1f} ms (processing; excludes display)')

    def show_error(self, message):
        self.summary.setPlainText('ERROR\n'+message)
        self.statusBar().showMessage('Failed; any displayed result is from an earlier run')

    def display_image(self, *args):
        if self.result is None:return
        key=self.view.currentText()
        is_stack=key in self.result.arrays and '/model_all_' in key
        self.model.setEnabled(is_stack or key=='segmenter/model_rejection_panels')
        if key=='segmenter/model_rejection_panels':
            if self.model.currentIndex()<0:return
            from ._rgbd_color_model import render_model_rejection_maps
            diagnostics={k.split('/',1)[1]:v for k,v in self.result.arrays.items() if k.startswith('segmenter/')}
            pixels=render_model_rejection_maps(diagnostics,self.result.arrays['segmenter/object_mask'],900,
                model_id=int(self.model.currentText()))
        elif is_stack:
            from .tuning_backend import array_image
            values=self.result.arrays[key]
            if not len(values):return
            pixels=array_image(values[max(0,self.model.currentIndex())])
        elif key in self.result.images:
            pixels=self.result.images[key]
        else:return
        raw=self.result.arrays.get(key)
        if is_stack and raw is not None:raw=raw[max(0,self.model.currentIndex())]
        overlay_ok=(raw is not None and raw.ndim==2 and raw.dtype==bool and
                    'color' in self.result.arrays and self.result.arrays['color'].shape[:2]==raw.shape)
        self.overlay.setEnabled(overlay_ok)
        from .tuning_backend import diagnostic_metadata
        metadata=diagnostic_metadata(self.result,key)
        note='Click for raw values. '
        if raw is not None and raw.ndim==2:
            if raw.dtype==bool:note+='White=true, black=false. '
            elif key.endswith(('_label_map','_group_map','/fusion_decision')):
                note+='Categorical colours: black=0; each colour is an ID, not a magnitude. Click for ID/meaning. '
            else:
                finite=raw[np.isfinite(raw)]
                note+=('Finite range: '+str(float(finite.min()))+' to '+str(float(finite.max()))+'. ' if finite.size else 'No finite values. ')
                note+='Per-view scaling; magenta=nonfinite. '
        if metadata:note+=json.dumps(metadata,default=str)
        if overlay_ok and self.overlay.isChecked():
            pixels=self.result.arrays['color'].copy()
            pixels[raw]=(.4*pixels[raw]+.6*np.array([255,30,30])).astype(np.uint8)
            note+=' Red=true, unchanged image=false; independent conditions may overlap.'
        self.view_note.setText(note)
        self.inspector.clear()
        pixels=np.ascontiguousarray(pixels,dtype=np.uint8)
        h,w=pixels.shape[:2]
        self._display_shape=(h,w)
        qimage=QtGui.QImage(pixels.data,w,h,pixels.strides[0],QtGui.QImage.Format.Format_RGB888).copy()
        pixmap=QtGui.QPixmap.fromImage(qimage)
        if self.fit.isChecked():pixmap=pixmap.scaled(self.scroll.viewport().size(),QtCore.Qt.AspectRatioMode.KeepAspectRatio,QtCore.Qt.TransformationMode.FastTransformation)
        self.canvas.setPixmap(pixmap)
        self.canvas.setMinimumSize(1,1) if self.fit.isChecked() else self.canvas.setMinimumSize(w,h)

    def inspect_canvas(self, label_x, label_y):
        if self.result is None or self.canvas.pixmap() is None:return
        pixmap=self.canvas.pixmap()
        if pixmap.isNull():return
        # QLabel centres the pixmap in its contents rectangle. Account for
        # letterboxing and native-size scrolling, not only fit-to-window views.
        rect=self.canvas.contentsRect()
        pw=pixmap.width()/pixmap.devicePixelRatio(); ph=pixmap.height()/pixmap.devicePixelRatio()
        left=rect.x()+(rect.width()-pw)/2; top=rect.y()+(rect.height()-ph)/2
        if not (left<=label_x<left+pw and top<=label_y<top+ph):return
        h,w=self._display_shape
        x=min(w-1,int((label_x-left)*w/pw)); y=min(h-1,int((label_y-top)*h/ph))
        from .tuning_backend import inspect_pixel
        try:
            info=inspect_pixel(self.result,self.view.currentText(),x,y,max(0,self.model.currentIndex()))
            self.inspector.setPlainText(json.dumps(info,indent=2,default=str))
        except ValueError as exc:self.inspector.setPlainText(str(exc))

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'view'):self.display_image()

    def project(self):
        crop=[s.value() for s in self.crop]
        return copy.deepcopy({'version':1,'paths':self.paths,'crop':crop,'config':{**self.config,'crop':crop}})

    def save_project(self):
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Save project','','JSON (*.json)')
        if path:
            try:Path(path).write_text(json.dumps(self.project(),indent=2,allow_nan=False)+'\n',encoding='utf-8')
            except Exception as exc:self.show_error(str(exc))

    def load_project(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Load project','','JSON (*.json)')
        if not path:return
        try:
            data=json.loads(Path(path).read_text(encoding='utf-8'))
            if data.get('version')!=1:raise ValueError('Unsupported project version')
            validate_project(data)
            self._building=True
            self.config=data['config']; self.config['crop']=list(data['crop']); self.paths=data['paths']
            for key,label in self.path_labels.items():label.setText(self.paths.get(key,''))
            for spin,value in zip(self.crop,data['crop']):spin.setValue(value)
            self.source.setCurrentText(self.config['source']);self.target.setCurrentText(self.config['target'])
            self.mode.setCurrentText(self.config.get('segmenter',{}).get('fusion_mode','union'))
            self.repair.setChecked(self.config.get('repair_enabled',False));self.details.setChecked(self.config.get('diagnostics',False))
            self.unit.setCurrentIndex(0 if self.config.get('depth_scale_m',.001)==.001 else (1 if self.config.get('depth_scale_m')==1 else 2))
            self._building=False;self.select_parameter();self.changed()
        except Exception as exc:self._building=False;self.show_error(str(exc))

    def export_result(self):
        if self.result is None and self.search_result is None:return
        directory=QtWidgets.QFileDialog.getExistingDirectory(self,'Export into a new subdirectory (continues if cancelled)')
        if not directory:return
        self.debounce.stop()
        self.runner.submit({'kind':'export','directory':directory,'result':self.result,
            'project':self.result_project or self.project(),'search_result':self.search_result})

    def edit_config(self):
        dialog=QtWidgets.QDialog(self);dialog.setWindowTitle('All pipeline settings (JSON)');dialog.resize(650,650)
        layout=QtWidgets.QVBoxLayout(dialog); editor=QtWidgets.QPlainTextEdit(json.dumps(self.config,indent=2));layout.addWidget(editor)
        buttons=QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok|QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons)
        if dialog.exec()!=QtWidgets.QDialog.DialogCode.Accepted:return
        try:
            config=json.loads(editor.toPlainText())
            if not isinstance(config,dict):raise ValueError('JSON object required')
            validate_project({**self.project(),'crop':config.get('crop',self.project()['crop']),'config':config})
            self.config=config
            self.sync_options();self.select_parameter();self.changed()
        except Exception as exc:self.show_error(str(exc))

    def sync_options(self):
        self._building=True
        for spin,value in zip(self.crop,self.config.get('crop',[0,0,640,400])):spin.setValue(value)
        self.source.setCurrentText(self.config['source']);self.target.setCurrentText(self.config['target'])
        self.mode.setCurrentText(self.config.get('segmenter',{}).get('fusion_mode','union'))
        self.repair.setChecked(self.config.get('repair_enabled',False));self.details.setChecked(self.config.get('diagnostics',False))
        self.unit.setCurrentIndex(0 if self.config.get('depth_scale_m',.001)==.001 else (1 if self.config.get('depth_scale_m')==1 else 2))
        self._building=False

    def batch_dialog(self):
        dialog=QtWidgets.QDialog(self);dialog.setWindowTitle('Batch evaluation / labelled parameter search');dialog.resize(650,450)
        layout=QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel('Manifest: registered image paths, optional ground_truth_mask / expected_count, train or validation split.'))
        row=QtWidgets.QHBoxLayout();manifest=QtWidgets.QLineEdit();choose=QtWidgets.QPushButton('Choose manifest')
        def pick():
            path,_=QtWidgets.QFileDialog.getOpenFileName(dialog,'Manifest','','JSON (*.json)')
            if path:manifest.setText(path)
        choose.clicked.connect(pick);row.addWidget(manifest);row.addWidget(choose);layout.addLayout(row)
        settings=QtWidgets.QHBoxLayout()
        mode=QtWidgets.QComboBox();mode.addItems(['batch','grid','random'])
        objective=QtWidgets.QComboBox();objective.addItems(['mask_iou','count_correct'])
        limit=QtWidgets.QSpinBox();limit.setRange(1,10000);limit.setValue(30)
        seed=QtWidgets.QSpinBox();seed.setRange(0,2147483647)
        for name,w in [('Mode',mode),('Objective',objective),('Max trials',limit),('Seed',seed)]:settings.addWidget(QtWidgets.QLabel(name));settings.addWidget(w)
        layout.addLayout(settings)
        layout.addWidget(QtWidgets.QLabel('Search space JSON: finite values per stage.parameter. Validation does not select parameters.'))
        space=QtWidgets.QPlainTextEdit('{"segmenter.background_chroma_tolerance": [15, 20, 25]}');layout.addWidget(space)
        buttons=QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok|QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons)
        if dialog.exec()!=QtWidgets.QDialog.DialogCode.Accepted:return
        try:
            self.config['crop']=[s.value() for s in self.crop]
            payload={'kind':'batch' if mode.currentText()=='batch' else 'search','manifest':manifest.text(),
                'config':self.config,'crop':[s.value() for s in self.crop],'method':mode.currentText(),
                'objective':objective.currentText(),'space':json.loads(space.toPlainText()),'max_trials':limit.value(),'seed':seed.value()}
            self.debounce.stop();self.runner.submit(payload)
        except Exception as exc:self.show_error(str(exc))

    def apply_best(self):
        if not self.search_result or not self.search_result.get('best'):return
        self.config=copy.deepcopy(self.search_result['best']['config'])
        self.sync_options();self.select_parameter();self.changed()

    def closeEvent(self,event):
        self.debounce.stop();self.runner.shutdown();event.accept()


def main():
    multiprocessing.freeze_support()
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window=TunerWindow();window.show()
    return app.exec()


if __name__=='__main__':
    raise SystemExit(main())
