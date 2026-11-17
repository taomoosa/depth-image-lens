"""Paged, exhaustive spatial diagnostics, rendered one page at a time."""
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
import numpy as np
from .debug import _opencv, depth_canvas
from .diagnostics import _atomic_write


def _json(value):
    if isinstance(value, dict): return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_json(v) for v in value]
    if isinstance(value, np.ndarray): return _json(value.tolist()) if value.size < 256 else {'shape': list(value.shape)}
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not math.isfinite(value): return str(value)
    if value is None or isinstance(value, (str, bool, int, float)): return value
    return str(value)


class _IncompleteRollback(OSError):
    """Recovery copies must be retained for manual restoration."""


def _owned_bundle_files(directory):
    """Read only a validated flat bundle manifest; never glob-delete pages."""
    index = directory / 'index.json'
    if not index.exists() and not index.is_symlink():
        return set()
    if index.is_symlink() or not index.is_file():
        raise ValueError('existing index must be a regular bundle manifest')
    try:
        data = json.loads(index.read_text(encoding='utf8'))
        if (not isinstance(data, dict) or data.get('version') != 1
                or not isinstance(data.get('result_type'), str)
                or not isinstance(data.get('entries'), list)
                or not isinstance(data.get('pages'), list)):
            raise ValueError('not a supported bundle index')
        names = []
        for page in data['pages']:
            name = page['file']
            if not isinstance(name, str) or not re.fullmatch(r'[0-9]{3,}_(overview|models)\.png', name):
                raise ValueError('invalid bundle page name')
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError('duplicate bundle page name')
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise ValueError('existing index is not a valid full-debug bundle; refusing overwrite') from exc
    owned = {'index.json', *names}
    for name in owned:
        path = directory / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError('bundle outputs must be regular files, not links or directories')
    return owned


def _publish_bundle(staged, destination, owned, backup):
    """Replace owned files only, with rollback for ordinary write failures."""
    new_names = {p.name for p in staged.iterdir()}
    for name in new_names:
        target = destination / name
        if target.is_symlink() or (target.exists() and name not in owned):
            raise FileExistsError(f'refusing to replace unrelated output: {name}')
        if target.exists() and not target.is_file():
            raise ValueError('bundle output is not a regular file')
    backup.mkdir()
    present = {name for name in owned if (destination / name).exists()}
    for name in present:
        shutil.copy2(destination / name, backup / name)
    touched = []
    try:
        # Publish index last; readers must not consume a bundle during an update.
        for name in sorted(new_names - {'index.json'}):
            os.replace(staged / name, destination / name)
            touched.append(name)
        for name in sorted(owned - new_names):
            if (destination / name).exists():
                (destination / name).unlink()
                touched.append(name)
        os.replace(staged / 'index.json', destination / 'index.json')
        touched.append('index.json')
    except Exception:
        rollback_errors = []
        for name in reversed(touched):
            try:
                if name in present:
                    os.replace(backup / name, destination / name)
                else:
                    (destination / name).unlink(missing_ok=True)
            except OSError as exc:
                rollback_errors.append(str(exc))
        if rollback_errors:
            raise _IncompleteRollback(f'export failed and rollback was incomplete; recovery files retained at {backup}')
        raise


def save_debug_bundle(result, directory, color=None, *, overwrite=False, **options):
    """Save full diagnostics, optionally replacing an existing bundle.

    Overwrite only manages files listed by a valid previous index. Other files
    are preserved, and name collisions are errors. Do not write concurrently.
    """
    if type(overwrite) is not bool:
        raise ValueError('overwrite must be bool')
    dest = Path(directory)
    if any(p.is_symlink() for p in [dest, *dest.parents]):
        raise ValueError('symlink output paths are not allowed')
    if not dest.parent.is_dir():
        raise ValueError('output parent must already exist')
    exists = dest.exists()
    if exists and (not overwrite or not dest.is_dir()):
        raise FileExistsError('output exists; use overwrite=True for a directory')
    owned = _owned_bundle_files(dest) if exists else set()
    work = Path(tempfile.mkdtemp(prefix='.depth-bundle-', dir=dest.parent))
    backup = work / 'backup'
    retain_recovery = False
    try:
        staged = work / 'rendered'
        _render_debug_bundle(result, staged, color, **options)
        if not exists:
            # Reserve the destination without replacing a directory created meanwhile.
            dest.mkdir(exist_ok=False)
        try:
            _publish_bundle(staged, dest, owned, backup)
        except Exception:
            if not exists and not any(dest.iterdir()):
                dest.rmdir()
            raise
    except _IncompleteRollback:
        retain_recovery = True
        raise
    finally:
        if not retain_recovery:
            shutil.rmtree(work)
    return dest / 'index.json'


def _render_debug_bundle(result, directory, color=None, *, columns=6, max_rows=4,
                      tile_width=640, models='all', color_order='RGB', depth=None,
                      depth_scale_m=.001):
    """Save all acquired spatial arrays to numbered PNG pages and index.json.

    A new directory is required. No diagnostic computation or detection is run.
    Models are integer diagnostic IDs, not zero-based stack positions. At most
    three model blocks share a page. Unsupported arrays are explicitly indexed.
    """
    for name, value, upper in [('columns', columns, 12), ('max_rows', max_rows, 12), ('tile_width', tile_width, 2048)]:
        if type(value) is not int or not (1 if name != 'tile_width' else 160) <= value <= upper:
            raise ValueError(f'{name} is outside the supported integer range')
    if columns * max_rows * tile_width ** 2 > 150_000_000:
        raise ValueError('requested page is too large; reduce columns, rows or tile width')
    if color_order not in ('RGB', 'BGR'): raise ValueError('explicit RGB or BGR required')
    content = vars(result)
    diagnostics = content.get('diagnostics', {}) or {}
    arrays = []
    absent = []
    def collect(value, prefix=''):
        if isinstance(value, dict):
            for key, item in value.items(): collect(item, (prefix + '/' + str(key)).strip('/'))
        elif isinstance(value, np.ndarray): arrays.append((prefix, value))
        elif value is None and prefix.endswith(('mask', 'map')): absent.append(prefix)
    collect(content)
    if color is not None:
        color = np.asarray(color)
        if color.dtype != np.uint8 or color.ndim != 3 or color.shape[-1] != 3:
            raise ValueError('color must be uint8 HxWx3')
        base = color if color_order == 'RGB' else color[..., ::-1]
    elif depth is not None:
        base = depth_canvas(depth, depth_scale_m)
    else:
        shape = next((a.shape for k,a in arrays if a.ndim == 2 and a.dtype == bool), None)
        if shape is None: raise ValueError('supply color or depth when result has no spatial mask')
        base = np.full((*shape, 3), 35, np.uint8)
    shape = base.shape[:2]
    if min(shape) < 2: raise ValueError('image must be at least 2x2')
    for key, array in arrays:
        if '/' not in key and array.ndim == 2 and array.dtype == bool and array.shape != shape:
            raise ValueError('color/result mask shape mismatch')
    if depth is not None and np.asarray(depth).shape != shape: raise ValueError('depth/source shape mismatch')
    metadata = diagnostics.get('diagnostic_metadata', {})
    metadata = metadata.get('maps', metadata) if isinstance(metadata, dict) else {}
    if not isinstance(metadata, dict): metadata = {}
    ids = np.asarray(diagnostics.get('model_diagnostic_ids', []))
    if isinstance(models, str):
        if models != 'all': raise ValueError("models must be 'all' or integer model IDs")
        selected_ids = set(int(i) for i in ids)
    else:
        try: values = list(models)
        except TypeError as exc: raise ValueError('models must be a sequence of integer IDs') from exc
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in values):
            raise ValueError('models must contain integer IDs')
        selected_ids = set(int(i) for i in values)
        if not selected_ids <= set(int(i) for i in ids): raise ValueError('unavailable model ID')
    entries, general, groups = [], [], {}
    def add(key, array=None, status='rendered', model_id=None, bit=None):
        field = key.split('/')[-1]
        info = dict(metadata.get(field, {})) if isinstance(metadata.get(field), dict) else {}
        parameters = content.get('parameters', {})
        known_units = {'model_all_color_distances': 'weighted Lab distance',
                       'model_color_distance': 'weighted Lab distance',
                       'model_background_distance': 'weighted Lab distance',
                       'model_all_luma_distances': 'OpenCV 8-bit Lab L distance (255/100 L*)',
                       'model_all_chroma_distances': 'Lab ab distance',
                       'model_all_source_distances_px': 'px (obstacle-free Chebyshev, not path length)',
                       'model_all_nearest_palette_indices': 'palette index; -1 = unavailable',
                       'model_all_rejection_flags': 'bit field; separate conditions below',
                       'background_combined_score': 'normalized squared Lab distance',
                       'background_chroma_distance': 'Lab ab distance',
                       'background_luma_distance': 'OpenCV 8-bit Lab L distance (255/100 L*)'}
        if field in known_units: info.setdefault('unit', known_units[field])
        threshold_parameter = {'model_all_color_distances': 'model_color_tolerance',
                               'model_color_distance': 'model_color_tolerance',
                               'depth_difference_m': 'depth_threshold_m'}.get(field)
        if threshold_parameter in parameters:
            info.setdefault('threshold', parameters[threshold_parameter])
            info.setdefault('parameter', threshold_parameter)
        if field == 'background_combined_score': info.setdefault('threshold', 1.0)
        if field == 'fusion_decision': info['legend'] = content.get('stages', {}).get('fusion', {}).get('decision_reasons', {})
        if not info.get('unit'):
            info['unit'] = 'm' if field.endswith('_m') else 'px' if field.endswith('_px') else 'not declared'
        entry = dict(key=key, status=status, metadata=_json(info))
        if model_id is not None: entry['model_id'] = model_id
        if bit is not None: entry.update(bit_mask=bit, rendered_dtype='bool')
        if array is not None: entry.update(shape=list(array.shape), dtype=str(array.dtype))
        entries.append(entry)
        descriptor = (entry, array, bit)
        if status == 'rendered':
            if model_id is None: general.append(descriptor)
            else: groups.setdefault(model_id, []).append(descriptor)
        return entry
    add('source', base)
    if depth is not None: add('source_depth_display', depth_canvas(depth, depth_scale_m))
    for key, a in arrays:
        field = key.split('/')[-1]
        if a.ndim == 2 and a.shape == shape and a.dtype.kind in 'buif': add(key, a)
        elif field.startswith('model_all_') and a.ndim == 3 and a.shape[1:] == shape and a.dtype.kind in 'buif':
            if len(a) != len(ids): raise ValueError('model stack and model_diagnostic_ids disagree')
            if not len(a): add(key, a, 'not_acquired')
            for i, model_id in enumerate(ids):
                status = 'rendered' if int(model_id) in selected_ids else 'omitted_by_model_filter'
                add(key, a[i], status, int(model_id))
                if field == 'model_all_rejection_flags' and status == 'rendered':
                    from ._rgbd_color_model import MODEL_REJECTION_FLAGS
                    for bit, reason in MODEL_REJECTION_FLAGS.items():
                        add(key + '/' + reason, a[i], model_id=int(model_id), bit=bit)
        else: add(key, a, 'unsupported_nonspatial_or_dtype')
    for key in absent: add(key, status='not_acquired')
    if not diagnostics: add('diagnostics', status='not_acquired')
    # Declared but absent maps are explicit, rather than silently synthesized.
    existing = {e['key'].split('/')[-1] for e in entries}
    for key in metadata:
        if key not in existing: add('diagnostics/' + key, status='not_acquired')
    th = max(2, round(shape[0]*tile_width/shape[1]))
    header = 120
    if columns*max_rows*tile_width*(th+header) > 150_000_000:
        raise ValueError('requested page is too large; reduce layout dimensions')
    dest = Path(directory)
    if any(p.is_symlink() for p in [dest, *dest.parents]): raise ValueError('symlink output paths are not allowed')
    if not dest.parent.is_dir(): raise ValueError('output parent must already exist')
    dest.mkdir(exist_ok=False)
    cv = _opencv()
    from .tuning_backend import diagnostic_image
    pages = []
    def wrap(text):
        lines, line = [], ''
        for char in text:
            if cv.getTextSize(line + char, cv.FONT_HERSHEY_SIMPLEX, .4, 1)[0][0] > tile_width - 10:
                lines.append(line); line = ''
            line += char
        if line: lines.append(line)
        return lines
    def render_page(items, ncols, kind):
        rows = math.ceil(len(items)/ncols)
        canvas = np.full((rows*(th+header), ncols*tile_width, 3), 24, np.uint8)
        filename = f'{len(pages)+1:03d}_{kind}.png'
        for slot, item in enumerate(items):
            if item is None: continue
            entry, a, bit = item
            key = entry['key']; field = key.split('/')[-1]
            a = (a & bit) != 0 if bit is not None else a
            rgb = diagnostic_image('bundle/' + field, a)
            finite = np.isfinite(a)
            note = 'White=true; black=false; independent masks may overlap.' if a.dtype == bool else ''
            categorical = field.endswith(('_label_map', '_group_map')) or field == 'fusion_decision'
            if categorical:
                entry['category_ids'] = [int(v) for v in np.unique(a)]
            if a.ndim == 2 and a.dtype != bool:
                numbers = a[finite]
                note = f'Finite [{float(numbers.min()):.5g}, {float(numbers.max()):.5g}]' if numbers.size else 'No finite samples'
                note += '; magenta=NaN; red=+Inf; blue=-Inf'
                if categorical: note = 'Categorical IDs, not magnitudes; black=0. IDs in index.'
                rgb[np.isposinf(a)] = (255, 0, 0); rgb[np.isneginf(a)] = (0, 100, 255)
                entry['finite_range'] = [float(numbers.min()), float(numbers.max())] if numbers.size else None
                entry['nan_pixels'] = int(np.isnan(a).sum())
            row, col = divmod(slot, ncols)
            entry.update(page=filename, row=row, column=col)
            x, y = col*tile_width, row*(th+header)
            canvas[y+header:y+header+th, x:x+tile_width] = cv.resize(rgb, (tile_width, th), interpolation=cv.INTER_NEAREST)
            title = (f"Model {entry['model_id']}: " if 'model_id' in entry else '') + key
            info = entry['metadata']
            lines = wrap(title)
            lines += wrap(note)
            detail = '; '.join(f'{k}={v}' for k,v in info.items() if k in ('unit','threshold','comparison'))
            if info.get('kind') == 'gate_margin':
                detail = 'margin boundary=0; ' + detail.replace('threshold=', 'original threshold=')
            lines += wrap(detail)
            for j, line in enumerate(lines[:6]):
                cv.putText(canvas, line, (x+5,y+17+j*17), cv.FONT_HERSHEY_SIMPLEX,.4,(235,235,235),1,cv.LINE_AA)
            if len(lines)>6: entry['caption_overflow'] = 'See complete metadata in index.json'
        ok, encoded = cv.imencode('.png', canvas[..., ::-1])
        if not ok: raise OSError('PNG encoding failed')
        _atomic_write(dest/filename, lambda stream: stream.write(encoded.tobytes()), overwrite=False)
        pages.append(dict(file=filename, width=int(canvas.shape[1]), height=int(canvas.shape[0]), kind=kind))
    capacity = columns*max_rows
    for start in range(0,len(general),capacity): render_page(general[start:start+capacity], columns, 'overview')
    # Two-column model blocks, aligned horizontally; continue long blocks on later pages.
    block_cols = min(2, columns)
    per_page = min(3, max(1, columns//block_cols))
    model_items = list(groups.values())
    for start in range(0,len(model_items),per_page):
        batch = model_items[start:start+per_page]
        block_cols = columns // len(batch)
        block_capacity = block_cols*max_rows
        for offset in range(0,max(map(len,batch)),block_capacity):
            chunks = [g[offset:offset+block_capacity] for g in batch]
            rows = max(math.ceil(len(c)/block_cols) for c in chunks)
            items = []
            for row in range(rows):
                for chunk in chunks:
                    for col in range(block_cols):
                        i = row*block_cols+col
                        items.append(chunk[i] if i<len(chunk) else None)
            render_page(items, block_cols*len(batch), 'models')
    payload = dict(version=1, result_type=type(result).__name__, source_shape=list(shape),
        layout=dict(columns=columns,max_rows=max_rows,tile_width=tile_width), pages=pages, entries=entries,
        parameters=_json(content.get('parameters', {})), stages=_json(content.get('stages', {})),
        diagnostic_metadata=_json(metadata),
        notes=['Display only; no detection recomputation. Missing diagnostics are not inferred.',
               'Numeric maps use finite per-panel scaling, not comparable brightness across maps.',
               'NaN=not evaluated or unavailable; consult each map metadata for scope. Zero is a real value.',
               'All overlapping model rejection bits are shown separately, including on selected pixels.',
               'Only one page and one diagnostic tile are rendered at a time; retained input stacks still consume memory.'])
    _atomic_write(dest/'index.json', lambda stream: stream.write((json.dumps(payload,indent=2,allow_nan=False)+'\n').encode()), overwrite=False)
    return dest/'index.json'
