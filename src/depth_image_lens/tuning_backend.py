"""Local, Qt-independent tuning workflows; no camera or network side effects."""
from dataclasses import dataclass
from copy import deepcopy
import inspect
import itertools
import json
from pathlib import Path
import time
import numpy as np
from .background import RGBDBackgroundSegmenter
from .mask_cleanup import cleanup_mask
from .object_count import OverheadObjectCounter
from .outline import OuterContourDetector
from .geometry import CameraIntrinsics

STAGES = {'segmenter': RGBDBackgroundSegmenter, 'repair': cleanup_mask,
          'counter': OverheadObjectCounter, 'outline': OuterContourDetector}
MODES = ('union', 'depth_required', 'evidence', 'color_seeded', 'depth_anchored',
         'color_supported', 'color_region', 'color_local', 'color_model')
ARRAY_ROLES = ('color', 'background_color', 'depth', 'background_depth', 'mask',
               'observed_mask', 'exclusion_mask', 'ground_truth_mask')


def parameter_defaults(stage):
    """Return constructor defaults, excluding externally supplied objects/arrays."""
    return {k: v.default for k, v in inspect.signature(STAGES[stage]).parameters.items()
            if v.default is not inspect.Parameter.empty
            and k not in ('background_segmenter', 'observed_mask', 'protected_mask')}


def default_config():
    return dict(version=1, source='segmenter', target='mask', depth_scale_m=.001,
                crop=[0, 0, 640, 400], diagnostics=False, repair_enabled=False,
                **{name: parameter_defaults(name) for name in STAGES})


def load_config(path):
    value = json.loads(Path(path).read_text(encoding='utf8'))
    if not isinstance(value, dict) or value.get('version') != 1:
        raise ValueError('expected configuration version 1')
    result = default_config()
    unknown = value.keys() - result.keys()
    if unknown:
        raise ValueError('unknown configuration keys: ' + ', '.join(sorted(unknown)))
    result.update(value)
    return result


def save_config(path, config):
    Path(path).write_text(json.dumps(config, indent=2, allow_nan=False) + '\n', encoding='utf8')


def load_array(path, role, npz_key=None):
    """Read RGB PNG/JPEG, lossless depth PNG, or non-pickle NumPy arrays."""
    import cv2
    if role not in ARRAY_ROLES:
        raise ValueError('unknown array role')
    path = Path(path)
    if path.suffix.lower() == '.npy':
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            key = npz_key or (role if role in archive.files else None)
            if key is None and len(archive.files) == 1:
                key = archive.files[0]
            if key not in archive.files:
                raise ValueError('select an explicit NPZ key for ' + role)
            array = archive[key].copy()
    else:
        if 'depth' in role and path.suffix.lower() != '.png':
            raise ValueError('depth requires lossless PNG, NPY or NPZ')
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise ValueError('cannot read ' + str(path))
        if 'color' in role and array.ndim == 3 and array.shape[2] == 3:
            array = array[..., ::-1].copy()
    if role.endswith('mask') or role == 'mask':
        if array.ndim != 2:
            raise ValueError('mask must be single channel')
        if not np.isfinite(array).all():
            raise ValueError('mask must be finite')
        array = array != 0
    return array


@dataclass
class PreparedInput:
    arrays: dict
    transform: dict
    intrinsics: object = None

    def to_source_xy(self, points, role=None):
        """Map pixel centres to a source role (depth by default when supplied)."""
        p = np.asarray(points, float)
        x, y, _, _ = self.transform['crop']
        scale = (self.transform['scale'] if role is None else
                 self.transform['sources'][role]['scale'])
        return (p + [x, y] + .5) / scale - .5


def prepare_inputs(arrays, crop=(0, 0, 640, 400), intrinsics=None):
    """Inputs must describe the same registered field of view at either resolution.

    Resize first, then crop in normalized 640x400 coordinates. Depth values and
    invalid sentinels are never scaled or interpolated. RGB uses area reduction;
    masks and depth use exact nearest-neighbour (pixel-centre convention).
    """
    import cv2
    arrays = {k: np.asarray(v) for k, v in arrays.items() if v is not None}
    if not arrays or arrays.keys() - set(ARRAY_ROLES):
        raise ValueError('known, nonempty input arrays required')
    if len(crop) != 4 or any(type(v) is not int for v in crop):
        raise ValueError('crop must be four integers: x, y, width, height')
    x, y, w, h = crop
    if x < 0 or y < 0 or w < 2 or h < 2 or x+w > 640 or y+h > 400:
        raise ValueError('crop must fit 640x400 with width and height >=2')
    out = {}
    sources = {}
    for key, value in arrays.items():
        shape = value.shape[:2]
        if shape not in ((400, 640), (800, 1280)):
            raise ValueError(f'{key}: input must be 640x400 or 1280x800')
        sources[key] = dict(source_shape=list(shape), scale=640 / shape[1])
        color = 'color' in key
        if color and (value.dtype != np.uint8 or value.shape != (*shape, 3)):
            raise ValueError('color arrays must be uint8 RGB HxWx3')
        if not color and (value.ndim != 2 or value.dtype.kind not in 'buif'):
            raise ValueError('depth/mask must be a numeric HxW array')
        if key.endswith('mask') or key == 'mask':
            if not np.isfinite(value).all():
                raise ValueError('mask must be finite')
            value = value != 0
        if shape == (800, 1280):
            if color:
                value = cv2.resize(value, (640, 400), interpolation=cv2.INTER_AREA)
            else:
                # Exact nearest centre samples, choosing the lower index at ties
                # (OpenCV INTER_NEAREST_EXACT convention), preserving every dtype.
                value = value[::2, ::2]
        out[key] = value[y:y+h, x:x+w].copy()
    # Intrinsics describe the current depth grid, not whichever dict key came first.
    source_role = next((key for key in ('depth', 'color', 'background_depth',
                       'background_color') if key in arrays), next(iter(arrays)))
    shape = sources[source_role]['source_shape']
    scale = sources[source_role]['scale']
    if intrinsics is not None:
        if isinstance(intrinsics, dict):
            intrinsics = CameraIntrinsics(**intrinsics)
        intrinsics = CameraIntrinsics(intrinsics.fx*scale, intrinsics.fy*scale,
            (intrinsics.cx+.5)*scale-.5-x, (intrinsics.cy+.5)*scale-.5-y)
    return PreparedInput(out, dict(source_shape=list(shape), normalized_shape=[400, 640],
        crop=list(crop), scale=scale, source_role=source_role, sources=sources,
        registration_assumption='same registered field of view; resizing does not perform registration',
        coordinates='cropped pixel centres; x right, y down'), intrinsics)


@dataclass
class PipelineResult:
    images: dict
    arrays: dict
    summary: dict
    object_mask: np.ndarray
    elapsed_ms: float


def array_image(array):
    """Render a diagnostic array to RGB; heatmaps use finite per-image scaling."""
    a = np.asarray(array)
    if a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3:
        return a.copy()
    if a.ndim != 2:
        raise ValueError('display array must be HxW or uint8 RGB')
    if a.dtype == bool:
        gray = a.astype(np.uint8)*255
    else:
        valid = np.isfinite(a)
        gray = np.zeros(a.shape, np.uint8)
        if valid.any():
            lo, hi = float(a[valid].min()), float(a[valid].max())
            if hi > lo:
                gray[valid] = np.clip((a[valid]-lo)/(hi-lo)*255, 0, 255).astype(np.uint8)
            elif lo != 0:
                # A constant nonzero map is not an empty/false map. Its exact
                # value is available in the inspector and finite range legend.
                gray[valid] = 160
    rgb = np.repeat(gray[..., None], 3, axis=2)
    if a.dtype != bool:
        rgb[~np.isfinite(a)] = [255, 0, 255]
    return rgb



def diagnostic_image(key, array):
    """Use categorical colours for label/decision maps, not ordered heatmaps."""
    a = np.asarray(array)
    if a.ndim == 2 and np.issubdtype(a.dtype, np.integer) and (
            key.endswith(('_label_map', '_group_map', '/fusion_decision'))):
        ids = a.astype(np.int64)
        # Stable hash colours; zero stays black. Labels are nominal IDs, not scores.
        packed = (ids * 2654435761) & 0xFFFFFF
        rgb = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], axis=-1).astype(np.uint8)
        rgb[ids == 0] = 0
        return rgb
    return array_image(a)


def diagnostic_metadata(result, key):
    """Return descriptive metadata without recomputing detector logic."""
    if '/' not in key:
        return {}
    stage, field = key.split('/', 1)
    metadata = result.summary.get('diagnostics', {}).get(stage, {}).get('diagnostic_metadata', {})
    if not isinstance(metadata, dict):
        return {}
    maps = metadata.get('maps', metadata)
    info = dict(maps.get(field, {})) if isinstance(maps, dict) and isinstance(maps.get(field, {}), dict) else {}
    stages = result.summary.get('stages', {}).get(stage, {})
    stages = stages.get('stages', stages.get('stage_results', {}))
    meanings = stages.get('fusion', {}).get('decision_reasons', {})
    if field == 'fusion_decision' and meanings:info['legend'] = meanings
    if field.startswith('reason_decision_') and field.endswith('_mask'):
        code = field[len('reason_decision_'):-len('_mask')]
        if code in meanings:info['decision_meaning'] = meanings[code]
    return info


def inspect_pixel(result, key, x, y, model_index=0):
    """Inspect retained raw arrays; overlapping reason masks remain independent."""
    if key not in result.arrays:
        raise ValueError('This view is a composite without a one-to-one raw array')
    values = result.arrays[key]
    model_id = None
    if '/model_all_' in key and values.ndim == 3:
        if not 0 <= model_index < len(values):
            raise ValueError('Model index is unavailable')
        values = values[model_index]
        ids = result.arrays.get('segmenter/model_diagnostic_ids', [])
        if model_index < len(ids):model_id = int(ids[model_index])
    if values.ndim not in (2, 3) or not (0 <= y < values.shape[0] and 0 <= x < values.shape[1]):
        raise ValueError('Pixel is outside the selected array')
    value = values[y, x]
    value = value.tolist() if isinstance(value, np.ndarray) else value.item()
    crop = result.summary.get('transform', {}).get('crop', [0, 0])
    info = dict(view=key, crop_xy=[x, y], normalized_xy=[x+crop[0], y+crop[1]],
                value=value, metadata=diagnostic_metadata(result, key))
    if model_id is not None:info['model_id'] = model_id
    stage = key.split('/', 1)[0] if '/' in key else 'segmenter'
    info['active_reasons'] = [name.split('/', 1)[1] for name, array in result.arrays.items()
        if name.startswith(stage+'/') and any(token in name.split('/', 1)[1] for token in ('reason_', 'region_fail_', '_blocked_mask', '_unusable_mask', '_invalid_mask', '_budget_exceeded_mask'))
        and array.dtype == bool and array.ndim == 2 and array.shape == values.shape[:2] and array[y, x]]
    info['reason_descriptions'] = {name: diagnostic_metadata(result, stage+'/'+name) for name in info['active_reasons']}
    stage_summary = result.summary.get('stages', {}).get(stage, {})
    stages = stage_summary.get('stages', stage_summary.get('stage_results', {}))
    decisions = stages.get('fusion', {}).get('decision_reasons', {})
    decision = result.arrays.get(stage+'/fusion_decision')
    if decision is not None and decision.shape == values.shape[:2]:
        code = int(decision[y, x]); info['fusion_decision'] = dict(code=code, meaning=decisions.get(str(code), 'see stage summary'))
    info['reasons_note'] = 'Independent conditions can overlap; these are not exclusive causes.'
    return info

def run_pipeline(prepared, config):
    """Run one immutable snapshot. Diagnostics affect recording, never decisions."""
    start = time.perf_counter()
    config = deepcopy(config)
    config['crop'] = list(prepared.transform['crop'])
    a = prepared.arrays
    if config['source'] not in ('segmenter', 'mask') or config['target'] not in ('mask', 'count', 'outline'):
        raise ValueError('invalid source or target')
    if 'color' not in a:
        raise ValueError('current color is required')
    scale = config['depth_scale_m']
    if isinstance(scale, bool) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('positive depth_scale_m required')
    diagnostic = config.get('diagnostics', False)
    if type(diagnostic) is not bool:
        raise ValueError('diagnostics must be bool')
    images = {k: array_image(v) for k,v in a.items()}
    arrays = {k: v.copy() for k,v in a.items()}
    summary = dict(transform=prepared.transform, configuration=config, stages={})
    observed = a.get('observed_mask', np.ones(a['color'].shape[:2], bool))
    uncertain = np.zeros(observed.shape, bool)
    protected = ~observed
    results = []
    if config['source'] == 'segmenter':
        required = ('depth', 'background_color', 'background_depth')
        if any(k not in a for k in required):
            raise ValueError('segmenter requires color, depth and both background images')
        result = RGBDBackgroundSegmenter(**config['segmenter']).detect(a['color'], a['depth'],
            background_color=a['background_color'], background_depth=a['background_depth'],
            color_order='RGB', registered_to_depth=True, depth_scale_m=scale,
            work_roi=observed, diagnostics=diagnostic)
        mask = result.object_mask
        observed = result.outline_observed_mask
        uncertain = result.uncertain_mask
        protected = ~observed
        results.append(('segmenter', result))
    else:
        if 'mask' not in a:
            raise ValueError('external mask source requires mask input')
        mask = a['mask'].copy()
    changed = np.zeros(mask.shape, bool)
    if config.get('repair_enabled'):
        result = cleanup_mask(mask, observed_mask=observed, protected_mask=protected,
                              **config['repair'])
        mask = result.object_mask
        changed = result.changed_mask
        results.append(('repair', result))
    if config['target'] == 'count':
        kwargs = {}
        if ('depth' in a) != ('background_depth' in a):
            raise ValueError('counter requires both depth images or neither')
        if 'depth' in a:
            kwargs.update(depth=a['depth'], background_depth=a['background_depth'])
        result = OverheadObjectCounter(background_segmenter=RGBDBackgroundSegmenter(**config['segmenter']),
            **config['counter']).detect_mask(a['color'], mask, color_order='RGB',
            background_color=a.get('background_color'), registered_to_depth=True,
            depth_scale_m=scale, observed_mask=observed, uncertain_mask=uncertain,
            cleanup_changed_mask=changed, exclusion_mask=a.get('exclusion_mask'), diagnostics=diagnostic, **kwargs)
        results.append(('counter', result))
    elif config['target'] == 'outline':
        kwargs = {}
        if prepared.intrinsics is not None and 'depth' in a:
            kwargs.update(depth=a['depth'], intrinsics=prepared.intrinsics, registered_to_mask=True)
        result = OuterContourDetector(**config['outline']).detect(mask, observed_mask=observed,
            depth_scale_m=scale, diagnostics=diagnostic, **kwargs)
        results.append(('outline', result))
    for stage, result in results:
        summary['stages'][stage] = result.to_dict()
        summary.setdefault('diagnostics', {})[stage] = {
            key: deepcopy(value) for key, value in getattr(result, 'diagnostics', {}).items()
            if not isinstance(value, np.ndarray)}
        for key,value in vars(result).items():
            if isinstance(value, np.ndarray):
                arrays[stage+'/'+key] = value
                if value.ndim == 2:
                    images[stage+'/'+key] = diagnostic_image(stage+'/'+key, value)
        for key,value in getattr(result, 'diagnostics', {}).items():
            if isinstance(value, np.ndarray):
                arrays[stage+'/'+key] = value
                if value.ndim == 2:
                    images[stage+'/'+key] = diagnostic_image(stage+'/'+key, value)
        if hasattr(result, 'render_debug'):
            images[stage+'/debug'] = result.render_debug(a['color'], color_order='RGB')
    images['result/mask'] = array_image(mask)
    overlay = a['color'].copy()
    overlay[mask] = (.6*overlay[mask]+.4*np.array([0, 255, 100])).astype(np.uint8)
    images['result/overlay'] = overlay
    elapsed = (time.perf_counter()-start)*1000
    summary['elapsed_ms'] = elapsed
    summary['display_note'] = ('Scalar maps use per-image min/max; constant nonzero maps are grey, zero is black, invalid values are magenta. Read raw values for comparisons. '
        'result/mask is the pipeline mask before counter rejection; inspect counter masks separately. '
        'exclusion_mask applies only to the counter. Crop coordinate mapping uses pixel centres.')
    return PipelineResult(images, arrays, summary, mask.copy(), elapsed)


def evaluate_dataset(samples, config, cancel=None):
    """Evaluate named prepared samples with optional ground-truth mask/count.

    samples entries: {'id', 'prepared', optional 'expected_count'}. Ground truth
    is never passed to detectors. Missing labels yield no objective score.
    """
    rows = []
    for sample in samples:
        if 'expected_count' in sample and (type(sample['expected_count']) is not int or sample['expected_count'] < 0):
            raise ValueError('expected_count must be a nonnegative integer')
        if cancel and cancel():
            break
        result = run_pipeline(sample['prepared'], config)
        row = dict(id=sample['id'], elapsed_ms=result.elapsed_ms, metrics={})
        truth = sample['prepared'].arrays.get('ground_truth_mask')
        if truth is not None:
            union = np.count_nonzero(truth | result.object_mask)
            row['metrics']['mask_iou'] = float(np.count_nonzero(truth & result.object_mask)/union) if union else 1.
        if 'expected_count' in sample:
            stage = result.summary['stages'].get('counter')
            if stage is not None:
                row['metrics']['count_correct'] = float(stage['count'] == sample['expected_count'])
                row['candidate_count'] = stage['candidate_count']
                row['count'] = stage['count']
        rows.append(row)
    return rows


def search_parameters(samples, config, space, *, method='grid', max_trials=50,
                      seed=0, objective='mask_iou', validation_samples=(), cancel=None, progress=None):
    """Bounded reproducible search; validation is evaluated only for the winner.

    space maps 'stage.parameter' to a finite list of values. Unlabelled data is
    rejected rather than treating self-consistency as ground-truth accuracy.
    """
    if method not in ('grid', 'random') or type(max_trials) is not int or not 1 <= max_trials <= 10000:
        raise ValueError('grid/random and max_trials in 1..10000 required')
    if objective not in ('mask_iou', 'count_correct'):
        raise ValueError('unsupported objective')
    if not samples or not space:
        raise ValueError('training samples and search space required')
    ids = [s['id'] for s in samples]
    vids = [s['id'] for s in validation_samples]
    if len(set(ids)) != len(ids) or len(set(vids)) != len(vids) or set(ids) & set(vids):
        raise ValueError('sample IDs must be unique; training and validation must be disjoint')
    for sample in samples:
        if objective == 'mask_iou' and 'ground_truth_mask' not in sample['prepared'].arrays:
            raise ValueError('mask IoU optimization requires a ground-truth mask per training sample')
        if objective == 'count_correct' and ('expected_count' not in sample or config['target'] != 'count'):
            raise ValueError('count objective requires expected_count and count target')
    keys = list(space)
    values = []
    for key in keys:
        stage, name = key.split('.', 1)
        if stage not in STAGES or name not in parameter_defaults(stage):
            raise ValueError('unknown search parameter: ' + key)
        active = (stage == 'segmenter' and (config['source'] == 'segmenter' or config['target'] == 'count')) or (stage == 'repair' and config.get('repair_enabled')) or (stage == 'counter' and config['target'] == 'count') or (stage == 'outline' and config['target'] == 'outline')
        if not active:
            raise ValueError('search stage is disabled: ' + stage)
        options = list(space[key])
        if not options:
            raise ValueError('empty parameter values')
        values.append(options)
    total = int(np.prod([len(v) for v in values], dtype=object))
    rng = np.random.default_rng(seed)
    if method == 'grid':
        combinations = itertools.islice(itertools.product(*values), max_trials)
    else:
        if total <= 100000:
            indices = rng.choice(total, size=min(total, max_trials), replace=False)
            def at(index):
                out = []
                for options in reversed(values):
                    out.append(options[int(index)%len(options)])
                    index //= len(options)
                return tuple(reversed(out))
            combinations = (at(i) for i in indices)
        else:
            raise ValueError('random search space exceeds 100000 combinations; narrow it')
    trials = []
    best = None
    for combination in combinations:
        if cancel and cancel():
            break
        candidate = deepcopy(config)
        changes = dict(zip(keys, combination))
        for key, value in changes.items():
            stage, name = key.split('.', 1)
            candidate[stage][name] = value
        try:
            rows = evaluate_dataset(samples, candidate, cancel=cancel)
            if len(rows) != len(samples):
                break
            score = float(np.mean([r['metrics'][objective] for r in rows]))
            trial = dict(parameters=changes, score=score, rows=rows)
            if best is None or score > best['score']:
                best = dict(config=candidate, score=score)
        except (ValueError, KeyError) as exc:
            trial = dict(parameters=changes, error=str(exc))
        trials.append(trial)
        if progress:
            progress(len(trials), min(total, max_trials), trial)
    validation = []
    if best is not None and not (cancel and cancel()):
        validation = evaluate_dataset(validation_samples, best['config'], cancel=cancel)
    return dict(trials=trials, best=best, validation=validation, objective=objective, seed=seed,
                cancelled=bool(cancel and cancel()), validation_used_for_selection=False)


def load_manifest(path):
    """Load a local JSON dataset manifest without executing embedded content.

    Format: {"version":1,"samples":[{"id":"scene", "split":"train",
    "arrays":{"color":"color.png","depth":{"path":"data.npz","key":"z"}},
    "expected_count":2}]}. Paths resolve relative to the manifest directory.
    """
    path = Path(path)
    document = json.loads(path.read_text(encoding='utf8'))
    if not isinstance(document, dict) or document.get('version') != 1 or not isinstance(document.get('samples'), list):
        raise ValueError('expected dataset manifest version 1 with samples list')
    result = []
    seen = set()
    for sample in document['samples']:
        name = sample.get('id')
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError('sample IDs must be unique nonempty strings')
        seen.add(name)
        split = sample.get('split', 'train')
        if split not in ('train', 'validation'):
            raise ValueError('split must be train or validation')
        arrays = {}
        for role, spec in sample['arrays'].items():
            if isinstance(spec, str):
                filename, key = spec, None
            elif isinstance(spec, dict):
                filename, key = spec['path'], spec.get('key')
            else:
                raise ValueError('array source must be path or {path,key}')
            arrays[role] = load_array(path.parent / filename, role, key)
        row = dict(id=name, split=split, arrays=arrays)
        if 'expected_count' in sample:
            count = sample['expected_count']
            if type(count) is not int or count < 0:
                raise ValueError('expected_count must be a nonnegative integer')
            row['expected_count'] = count
        if 'intrinsics' in sample:
            row['intrinsics'] = sample['intrinsics']
        result.append(row)
    return result
