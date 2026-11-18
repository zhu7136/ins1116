import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy import sparse
from per_pose_foot_guard import PerPoseGuard
import wholebody_solver as S
from test_wholebody_solver import SyntheticCertificateGuard
class GuardAdapterTests(unittest.TestCase):
    def setUp(self):
        shapes=[SimpleNamespace(kind='box',link=s+'_ankle_roll_link',parameters={'size':[.08,.08,.08]},local=np.eye(4)) for s in ['left','right']]
        self.body=SimpleNamespace(shapes=shapes,tri=np.array([[[.1,-2,-2],[.1,2,-2],[.1,0,2]]]))
        self.guard=PerPoseGuard(self.body);self.q=np.zeros((2,29));self.r=np.tile(np.eye(4),(2,1,1));self.frames=np.array([80,81])
        class Model:
            def fk(me,q):
                p=np.tile(np.eye(4),(len(q),1,1));p[:,0,3]=q[:,0]
                return {s+'_ankle_roll_link':p for s in ['left','right']},{}
        self.model=SimpleNamespace(model=Model())
    def test_original_both_boxes_new_wall_frame_rejected(self):
        old=self.guard.scan(self.model,self.q,self.r,self.frames);self.q[1,0]=.1;new=self.guard.scan(self.model,self.q,self.r,self.frames)
        passed,detail=self.guard.compare(old,new);self.assertFalse(passed)
        for side in ['left','right']:self.assertEqual(detail[side]['new_hit_poses'],1);self.assertEqual(new[side]['pose_ids'],[side+':80',side+':81'])
    def test_frame_binding_required(self):
        with self.assertRaises(ValueError):self.guard.scan(self.model,self.q,self.r,np.array([80,80]))
    def test_equal_recheck_passes(self):
        old=self.guard.scan(self.model,self.q,self.r,self.frames);self.assertTrue(self.guard.compare(old,old)[0])
    def test_actual_solver_assignment_after_perpose_rejection(self):
        class FakeBody:
            margin=.001;shapes=[SimpleNamespace(link='A'),SimpleNamespace(link='B')]
            def evaluate(me,model,q,roots,jacobian=False):
                g=np.array([-.01+roots[0,0,3]]);record=[dict(kind='self',frame=0,first=0,second=1)]
                if jacobian:return g,sparse.csr_matrix(([1.],([0],[0])),shape=(1,len(q)*35)),record,{}
                return g,record,{}
        class FakePerPose:
            def __init__(me,body):pass
            def scan(me,model,q,roots,frames):return float(roots[0,0,3])
            def compare(me,old,new):return new<=.001,{'TEST_STUB':dict(reject=new>.001)}
        class Model:
            def evaluate(me,q,roots,jacobian=False):
                v=np.zeros((len(q),90));return (v,np.zeros((len(q),90,35))) if jacobian else v
        d=np.zeros((11,35));d[:,0]=.01;lp=SimpleNamespace(success=True,x=d.ravel(),status=0,nit=1)
        q=np.zeros((997,29));roots=np.tile(np.eye(4),(997,1,1));rows=[]
        with patch.object(S,'WholeBodyTerms',return_value=FakeBody()),patch.object(S,'plan',return_value=[(0,10)]),\
             patch.object(S,'hard_rows',return_value=(sparse.csr_matrix((1,385)),np.zeros(1))),\
             patch.object(S,'joint_feasibility',return_value={'violation':0.}),\
             patch.object(S,'temporal_system',return_value=(np.zeros(1),sparse.csr_matrix((1,385)))),\
             patch.object(S,'temporal_values',return_value=np.zeros(1)),patch.object(S,'linprog',return_value=lp),\
             patch('per_pose_foot_guard.PerPoseGuard',FakePerPose):
            oq,oroot,report=S.solve(Model(),q,roots,np.zeros((997,90)),rows.append,enforce_per_pose_guard=True,certificate_guard=SyntheticCertificateGuard())
        self.assertTrue(rows[0]['accepted']);self.assertEqual(rows[0]['alpha'],.0625)
        self.assertEqual(len(rows[0]['backtracking']),5)
        self.assertTrue(all(not r['per_pose_guard_passed'] for r in rows[0]['backtracking'][:-1]))
        self.assertTrue(rows[0]['backtracking'][-1]['per_pose_guard_passed'])
        np.testing.assert_allclose(oroot[:11,0,3],.000625);np.testing.assert_array_equal(oq,q)
if __name__=='__main__':unittest.main(verbosity=2)