"""Fixed synthetic shapes and latent labels; no parameter-based relabeling."""
import numpy as np


def cases():
    import cv2
    rectangle=np.array([[50,40],[169,40],[169,139],[50,139]],float)
    for name in ('flat','rotated','concave','rounded','pattern','illumination_change',
                 'internal_fold','invalid_depth','mixed_boundary','registration_shift',
                 'cropped','hidden_aligned_layers','noise'):
        polygon=rectangle.copy();truth=polygon.copy();mask=np.zeros((180,220),np.uint8)
        if name=='rotated':
            a=np.deg2rad(17);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            polygon=np.rint((polygon-[110,90])@R.T+[110,90]);truth=polygon.copy()
        if name=='concave':
            polygon=np.array([[50,40],[169,40],[169,80],[100,80],[100,139],[50,139]])
            truth=polygon[[0,1,2,4,5]]
        if name=='cropped':polygon[:,0]-=65;truth=polygon[polygon[:,0]>0]
        if name=='rounded':
            cv2.ellipse(mask,(110,90),(60,50),0,0,360,1,-1);truth=np.zeros((0,2))
        else:cv2.fillPoly(mask,[polygon.astype(np.int32)],1)
        mask=mask.astype(bool);bg=np.full((*mask.shape,3),70,np.uint8)
        rgb=bg.copy();rgb[mask]=[30,80,200]
        depth=np.where(mask,980.,1000.)
        yy,xx=np.indices(mask.shape)
        overlap=np.zeros((0,2));hidden=np.zeros((0,2))
        if name=='pattern':rgb[mask&((xx//7+yy//7)%2==0)]=[230,210,20]
        if name=='illumination_change':rgb=np.clip(rgb.astype(int)+45,0,255).astype(np.uint8)
        if name=='internal_fold':
            depth[mask&(xx<65)]=920.;overlap=np.array([[50,40],[50,139]])
        if name=='invalid_depth':depth[30:70,40:80]=0
        if name=='mixed_boundary':
            interior=cv2.erode(mask.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool)
            depth[mask&~interior]=1040.
        if name=='registration_shift':depth=np.roll(depth,9,axis=1)
        if name=='hidden_aligned_layers':hidden=rectangle.copy()
        if name=='noise':depth+=np.random.default_rng(7721).normal(0,1,depth.shape)
        yield {'name':name,'color':rgb,'background':bg,'depth':depth,'mask':mask,
               'visible_corners':truth,'observable_overlap_corners':overlap,'hidden_overlap_corners':hidden}
