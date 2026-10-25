"""Frozen synthetic derivatives for the first diagnostics/fix comparison."""
import numpy as np
from .test_bridge_drape import scene

specs=[('flat','translate',1.,0.,True,False),('flat','scale',1.4,0.,False,False),('flat','roll',1.,8.,False,False),('u','translate',1.,0.,True,False),('u','scale',1.4,0.,False,False),('u','roll',1.,-8.,False,False),('slope','translate',1.,0.,False,False),('slope','scale',1.4,0.,False,False),('slope','roll',1.,8.,False,False),('separate','translate',1.,0.,True,False),('separate','scale',1.4,0.,False,False),('separate','missing',1.,0.,False,True)]
def cases():
 for kind,name,scale,angle,speck,missing in specs:
  m=scene(kind) if kind!='slope' else np.zeros((100,120),bool)
  if kind=='slope':
   for x in range(20,101):
    y=20+round((x-20)*.1);m[y:y+7,x]=True
  theta=np.deg2rad(angle);c,s=np.cos(theta),np.sin(theta);down=(-s,c)
  yy,xx=np.indices((190,220));dx=xx-30;dy=yy-28
  ox=np.rint((c*dx+s*dy)/scale).astype(int);oy=np.rint((-s*dx+c*dy)/scale).astype(int)
  inside=(ox>=0)&(ox<120)&(oy>=0)&(oy<100);mask=np.zeros_like(inside);mask[inside]=m[oy[inside],ox[inside]]
  if speck:mask.flat[np.argmin(xx*down[0]+yy*down[1])]=True
  depth=np.where(mask,1.5,2.);diff=2.-depth
  if missing:depth[:,127:133]=0
  shift=30*down[0]+28*down[1]
  yield {'name':kind+'_'+name,'depth':depth,'difference':diff,'options':{'upper_band':(shift+8*scale,shift+20*scale),'min_span_px':30*scale,'max_height_delta_px':12*scale,'down':down},'bridge_GT':'no' if kind=='separate' else 'yes','synthetic_derivative':True}
