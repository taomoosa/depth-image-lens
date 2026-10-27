"""Shared optional debug views for measured result objects; no detection changes."""
from pathlib import Path
import numpy as np
from .diagnostics import _atomic_write


def _opencv():
    try: import cv2
    except ImportError as exc: raise ImportError("Debug rendering needs the optional debug OpenCV extra") from exc
    return cv2


def depth_canvas(depth, depth_scale_m=.001):
    """Percentile display only; missing optical-axis depth is magenta."""
    from .depth import _validate_depth_scale, _numeric_image
    _validate_depth_scale(depth_scale_m)
    with np.errstate(over='ignore',invalid='ignore'):
        z=_numeric_image(depth,'depth')*depth_scale_m
    valid=np.isfinite(z)&(z>0);gray=np.zeros(z.shape,np.uint8)
    if valid.any():
        lo,hi=np.quantile(z[valid],[.02,.98])
        gray[valid]=128 if hi<=lo else np.clip(40+180*(z[valid]-lo)/(hi-lo),40,220).astype(np.uint8)
    rgb=np.repeat(gray[...,None],3,axis=2);rgb[~valid]=(170,30,170)
    return rgb


def _summary_lines(value, prefix="", limit=32):
    lines=[]
    if isinstance(value,dict):
        for key,item in value.items():
            if key.startswith('_') or key in ('diagnostics','source','inputs'): continue
            label=(prefix+'.'+key).strip('.')
            if isinstance(item,dict): lines.extend(_summary_lines(item,label,limit-len(lines)))
            elif isinstance(item,(str,bool,int,float,np.number)) or item is None:
                lines.append(f"{label}: {item}")
            elif isinstance(item,(tuple,list)) and len(item)<=5 and all(isinstance(x,(str,bool,int,float)) or x is None for x in item):
                lines.append(f"{label}: {item}")
            if len(lines)>=limit:break
    return lines[:max(limit,0)]


class DebugResultMixin:
    """Uniform explicit rendering on results; raw arrays are never mutated."""
    def render_debug(self, color=None, *, color_order="RGB", depth=None,
                     depth_scale_m=.001, mask_alpha=.35, panel=True, max_lines=32):
        cv2=_opencv()
        if color_order not in ('RGB','BGR'):raise ValueError('explicit RGB or BGR required')
        if type(panel) is not bool or type(max_lines) is not int or not 1<=max_lines<=100:
            raise ValueError('panel must be bool; max_lines integer 1..100')
        if not np.isscalar(mask_alpha) or not np.isfinite(mask_alpha) or not 0<=mask_alpha<=1:
            raise ValueError('mask_alpha must be finite in [0,1]')
        from .depth import _validate_depth_scale,_numeric_image
        _validate_depth_scale(depth_scale_m)
        layers=[]
        def collect(value,prefix=''):
            if isinstance(value,dict):
                for key,item in value.items():collect(item,(prefix+'.'+key).strip('.'))
            elif isinstance(value,np.ndarray) and value.ndim==2 and value.dtype==bool:
                layers.append((prefix,value))
        content=vars(self);collect(content)
        shapes={v.shape for _,v in layers}
        if len(shapes)>1:raise ValueError('result mask shapes disagree')
        shape=next(iter(shapes),None)
        invalid=None
        if color is not None:
            a=np.asarray(color)
            if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[-1]!=3 or min(a.shape[:2])<2:
                raise ValueError('color must be uint8 HxWx3')
            if shape is not None and a.shape[:2]!=shape:raise ValueError('color/result shape mismatch')
            shape=a.shape[:2];base=np.array(a if color_order=='RGB' else a[...,::-1],copy=True,order='C')
        elif depth is not None:
            base=depth_canvas(depth,depth_scale_m)
            if shape is not None and base.shape[:2]!=shape:raise ValueError('depth/result shape mismatch')
            shape=base.shape[:2]
        elif shape is not None:base=np.full((*shape,3),35,np.uint8)
        else:raise ValueError('supply color or depth when result contains no spatial diagnostics')
        palette=[(30,210,90),(240,150,40),(50,160,245),(235,75,90),(180,100,220),(50,220,220)]
        # Independent tiles preserve overlapping masks instead of painting one over another.
        tiles=[('Source (depth display is percentile-scaled)' if color is None and depth is not None else 'Source',base)]
        diagnostic=getattr(self,'diagnostics',{})
        if isinstance(diagnostic,dict):
            for name,points in diagnostic.items():
                if name.endswith('_xy') and isinstance(points,np.ndarray) and points.ndim==2 and points.shape[1]==2:
                    tile=base.copy()
                    for x,y in points:
                        if np.isfinite([x,y]).all() and 0<=x<shape[1] and 0<=y<shape[0]:
                            cv2.circle(tile,(round(float(x)),round(float(y))),1,(250,160,30),-1)
                    tiles.append((name+' (recorded samples; no gap filling)',tile))
        for i,(name,mask) in enumerate(layers):
            tile=base.copy();ink=(210,70,210) if any(k in name for k in ('unknown','uncertain','invalid')) else (240,150,40) if 'exclud' in name else palette[i%len(palette)]
            tile[mask]=np.rint((1-mask_alpha)*tile[mask]+mask_alpha*np.array(ink)).astype(np.uint8)
            tiles.append((name+f' ({int(mask.sum())} px)',tile))
        h,w=shape
        # Limit thumbnails, not their underlying arrays/measurements.
        scale=min(1.,480/max(h,w));th,tw=max(2,round(h*scale)),max(2,round(w*scale))
        cell_w=max(280,tw)
        shown=tiles[:13];ncols=min(3,len(shown));nrows=(len(shown)+ncols-1)//ncols
        canvas=np.full((nrows*(th+30),ncols*cell_w,3),24,np.uint8)
        for i,(name,tile) in enumerate(shown):
            y=(i//ncols)*(th+30);x=(i%ncols)*cell_w
            canvas[y+30:y+30+th,x:x+tw]=cv2.resize(tile,(tw,th),interpolation=cv2.INTER_NEAREST)
            cv2.putText(canvas,name[:max(12,cell_w//7)],(x+4,y+20),cv2.FONT_HERSHEY_SIMPLEX,.4,(235,235,235),1,cv2.LINE_AA)
        if panel:
            import textwrap
            lines=[type(self).__name__+' / measured evidence, not physical proof',
                'Tiles may be downscaled for display; parameters use processing pixels.',
                'No recorded masks: limited view.' if not layers else 'Each overlay uses a separate tile; missing diagnostics are not inferred.']
            summary=content.get('summary',{})
            # Show both sides before bounded detail traversal can exhaust the panel.
            for side,values in content.get('sides',summary.get('sides',{})).items():
                if isinstance(values,dict):
                    for key in ('estimate','status','reason','reasons','physical_grasp'):
                        if key in values:lines.append(f'{side}.{key}: {values[key]}')
            for key in ('parameters','thresholds'):
                value=content.get(key,content.get('summary',{}).get(key,{}))
                if isinstance(value,dict):lines+=_summary_lines(value,key,limit=16)
            lines+=_summary_lines(content,limit=max_lines)
            regions=content.get('regions',[])
            if regions:
                selected=sorted(regions,key=lambda v:v.get('shape_status')=='locally_planar')[:3]
                lines.append('Local cell detail: first 3 nonflat/unknown cells preferred; all in JSON.')
                for cell in selected:
                    lines+=_summary_lines({k:cell[k] for k in ('id','component_id','shape_status','valid_fraction','residual_p95_m','step_pairs','reasons') if k in cell},'cell',limit=7)
            diagnostic=getattr(self,'diagnostics',{})
            if isinstance(diagnostic,dict):lines+=_summary_lines(diagnostic,limit=8)
            if len(tiles)>len(shown):lines.append(f'{len(tiles)-len(shown)} further masks omitted from this view.')
            wrapped=[s for line in lines for s in textwrap.wrap(line,70)[:3]]
            out=np.full((max(canvas.shape[0],25+17*len(wrapped)),canvas.shape[1]+560,3),24,np.uint8)
            out[:canvas.shape[0],:canvas.shape[1]]=canvas
            for i,line in enumerate(wrapped):cv2.putText(out,line,(canvas.shape[1]+10,22+i*17),cv2.FONT_HERSHEY_SIMPLEX,.4,(230,230,230),1,cv2.LINE_AA)
            canvas=out
        return canvas if color_order=='RGB' else np.ascontiguousarray(canvas[...,::-1])

    def save_debug(self,path,color=None,*,color_order='RGB',overwrite=False,**options):
        if type(overwrite) is not bool:raise ValueError('overwrite must be bool')
        dest=Path(path)
        if dest.suffix.lower()!='.png':raise ValueError('debug destination must end in .png')
        if dest.parent.is_symlink() or not dest.parent.is_dir():raise ValueError('existing non-symlink parent directory required')
        a=self.render_debug(color,color_order=color_order,**options)
        ok,encoded=_opencv().imencode('.png',a[...,::-1] if color_order=='RGB' else a)
        if not ok:raise OSError('PNG encoding failed')
        _atomic_write(dest,lambda f:f.write(encoded.tobytes()),overwrite=overwrite)
        return dest
