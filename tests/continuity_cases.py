import numpy as np

def cases():
    shape=(90,150);base=np.zeros(shape,bool);base[23:27,18:133]=True
    rois={'left':(13,10,24,24),'right':(127,10,138,24)}
    out=[]
    def add(name,mask,truth,gap=0,alter=None):
        z=np.full(shape,1500,np.uint16);d=mask.astype(np.uint16)*500
        if alter:alter(z,d)
        out.append({'name':name,'depth':z,'difference':d,'truth':truth,'gap':gap,'rois':rois})
    add('thin_bridge',base,'yes')
    m=base.copy();m[:,75]=False
    add('small_gap_disabled',m,'no')
    add('small_gap_continuous_enabled',m,'yes',1)
    add('small_gap_background_depth',m,'no',1,lambda z,d:z.__setitem__((slice(None),75),2000))
    add('small_gap_missing_depth',m,'unknown',1,lambda z,d:z.__setitem__((slice(None),75),0))
    m=base.copy();m[:,72:79]=False;add('large_zero_gap',m,'no',2)
    add('adjacent_depth_step',base,'no',0,lambda z,d:z.__setitem__((slice(None),slice(75,None)),1100))
    m=np.zeros(shape,bool);m[23:67,18:24]=True;m[23:67,127:133]=True;add('independent_drapes',m,'no')
    m=np.zeros(shape,bool);m[21:24,12:139]=True;m[30:34,19:132]=True
    add('multiple_first_discontinuous',m,'yes',0,lambda z,d:z.__setitem__((slice(21,24),slice(75,None)),1100))
    m=np.zeros(shape,bool);m[23:27,18:75]=True;m[27:31,75:133]=True
    add('diagonal_only_depth_step',m,'no',0,lambda z,d:z.__setitem__((slice(None),slice(75,None)),1100))
    return out
