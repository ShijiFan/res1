import numpy as np
from pathlib import Path
import make_dense_map as m
with np.load(m.RUN/'reference_and_masks.npz') as z:
 y,omega,c2=[z[k].copy() for k in ('y','omega','c2')]
with np.load(m.CUBE) as z:vv=z['t37_d1_vv'][250:375,625:750];coh=z['t37_g12'][250:375,625:750]
base=np.load(m.RUN/'I28_target_t37_block_205.npy');aug=np.load(m.RUN/'I28_G29_target_t37_block_205.npy')
m.plot(vv,coh,y,omega,c2,base,aug)
