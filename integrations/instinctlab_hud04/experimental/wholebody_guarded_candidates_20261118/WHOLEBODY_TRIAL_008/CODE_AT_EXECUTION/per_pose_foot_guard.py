"""Pin and reuse concurrent86f05 guard; adds no collision exemptions."""
import hashlib,importlib.util
from pathlib import Path
import numpy as np
EXPECTED='f69a24efb4ccf34e82c53d9c45fb71b1dca6555aba97663bca920d38b6d0a797'
class PerPoseGuard:
    def __init__(self,body):
        here=Path(__file__).parent
        candidates=[here.parent/'nonlinear_guard_20261118/box_triangle_guard.py',here/'REMOTE_ADVANCE_REVIEW_86f05/nonlinear_guard_20261118/box_triangle_guard.py']
        paths=[p for p in candidates if p.is_file()]
        if not paths:raise ValueError('Pinned concurrent guard not recovered')
        path=paths[0]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED:raise ValueError('Concurrent guard changed; review before running')
        spec=importlib.util.spec_from_file_location('pinned86f05_guard',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.module=module;self.guard=module.BoxTriangleGuard(body.tri,tolerance=1e-9);self.feet={}
        for side in ['left','right']:
            shapes=[s for s in body.shapes if s.link==side+'_ankle_roll_link' and s.kind=='box']
            if len(shapes)!=1:raise ValueError('Exactly one original nominal footbox required')
            self.feet[side]=shapes[0]
    def scan(self,model,q,roots,frame_indices):
        frames=np.asarray(frame_indices)
        if frames.shape!=(len(q),) or not np.issubdtype(frames.dtype,np.integer) or len(np.unique(frames))!=len(frames):raise ValueError('Explicit native ordered frame binding')
        local,_=model.model.fk(q);result={}
        for side,shape in self.feet.items():
            poses=roots@local[shape.link]@shape.local
            result[side]=self.guard.scan(poses,np.array(shape.parameters['size'])/2,pose_ids=[side+':'+str(int(f)) for f in frames])
        return result
    def compare(self,old,new):
        result={side:self.module.reject_new_or_worse(old[side],new[side],numerical_slack=1e-12) for side in self.feet}
        return not any(r['reject'] for r in result.values()),{side:dict(reject=r['reject'],new_hit_poses=len(r['new_hit_pose_indices']),worsened_poses=len(r['worsened_pose_indices'])) for side,r in result.items()}