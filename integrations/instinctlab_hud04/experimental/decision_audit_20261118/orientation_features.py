"""Target URDF-frame orientation observables, NOT calibrated contact frames.

No source/target alignment is supplied or assumed. These features must not be
turned into a tracking objective until explicit frame correspondences exist.
"""
import numpy as np

LINKS=('waist_pitch_link','left_wrist_roll_link','right_wrist_roll_link',
       'left_ankle_roll_link','right_ankle_roll_link')

def evaluate(model,q,jacobian=True):
    if model.kind!='target' or model.anchor!=model.root:
        raise ValueError('Pinned target fixed-root URDF diagnostic only')
    pose,axes=model.fk(q)
    basis=np.stack([pose[link][:,:3,:3] for link in LINKS],axis=1)
    if not jacobian:return basis
    derivative=np.zeros((len(basis),len(LINKS),3,3,29))
    for feature,link in enumerate(LINKS):
        for c,name in enumerate(model.names):
            if name in model.ancestors[link]:
                _,axis=axes[name]
                for col in range(3):
                    derivative[:,feature,:,col,c]=np.cross(axis,basis[:,feature,:,col])
    return basis,derivative