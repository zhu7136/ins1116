"""Software tests. Synthetic fixtures are NOT motion acceptance."""
import copy,hashlib,os,sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from anatomical_directions import DirectionModel,pairs,direction_statistics
from direction_solver import joint_inequalities,solve

BASE=Path('/workspace/hud04-recovery-bf4a057')
TARGET=BASE/'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
SOURCE=Path('/workspace/hud04-recovery-fd12fe1/upstream/source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf')
class Gate:
    def __init__(self):self.count=0
    def step(self):self.count+=1

class DirectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(c):
        c.t=DirectionModel(TARGET,'target');c.s=DirectionModel(SOURCE,'source')
        c.q=(c.t.lower+c.t.upper)/2;c.seed=np.tile(c.q,(997,1))
    def test_unit_shapes(self):
        u,j=self.t.evaluate(self.seed[:2],True);self.assertEqual(u.shape,(2,8,3));self.assertEqual(j.shape,(2,8,3,29));np.testing.assert_allclose(np.linalg.norm(u,axis=2),1,atol=1e-12)
    def test_all_29_jacobian_columns(self):
        q=self.seed[:1].copy();u,j=self.t.evaluate(q,True)
        for col in range(29):
            d=np.zeros_like(q);d[:,col]=1e-6
            np.testing.assert_allclose((self.t.evaluate(q+d)-self.t.evaluate(q-d))/(2e-6),j[:,:,:,col],atol=2e-9,rtol=0)
    def test_target_against_prior_independent_fk(self):
        sys.path.insert(0,'/workspace/hud04-window-system-20261117')
        from named_target import NamedTarget
        prior=NamedTarget(TARGET);pose,_=self.t.fk(self.seed[:1]);q=self.t.order(self.seed[:1],self.t.names)[:,[self.t.names.index(n) for n in prior.names]]
        points,_=prior.evaluate(q[0],np.eye(4))
        for i,(_,link,offset) in enumerate(prior.features):np.testing.assert_allclose(points[i],pose[link][0,:3,:3]@offset+pose[link][0,:3,3],atol=2e-14,rtol=0)
    def test_source_against_prior_independent_fk(self):
        sys.path.insert(0,str(BASE/'integrations/instinctlab_hud04'))
        from kinematic_diagnostics import RobotTree
        old=RobotTree.read(SOURCE);q=np.zeros((2,29));q[1]=np.linspace(-.2,.2,29)
        actual,_=self.s.fk(q);expected=old.fk(q,self.s.names)
        for _,a,b in self.s.pairs:
            np.testing.assert_allclose(actual[a],expected[a],atol=2e-14,rtol=0);np.testing.assert_allclose(actual[b],expected[b],atol=2e-14,rtol=0)
    def test_source_waist_does_not_rotate_pelvis_local_legs(self):
        q=np.zeros((2,29))
        for name in ('waist_pitch_joint','waist_roll_joint','waist_yaw_joint'):q[1,self.s.names.index(name)]=.3
        u=self.s.evaluate(q);np.testing.assert_allclose(u[0,[0,1,4,5]],u[1,[0,1,4,5]],atol=1e-12,rtol=0)
    def test_uniform_model_scale_invariance(self):
        scaled=copy.deepcopy(self.s);scaled.joints=[]
        for name,typ,a,b,t,axis in self.s.joints:
            t=t.copy();t[:3,3]*=3.7;scaled.joints.append((name,typ,a,b,t,axis))
        q=np.tile(np.linspace(-.2,.2,29),(2,1));np.testing.assert_allclose(self.s.evaluate(q),scaled.evaluate(q),atol=1e-13,rtol=0)
    def test_common_world_transform_cancels(self):
        q=np.linspace(-.2,.2,29)[None];pose,_=self.s.fk(q);w=np.eye(4);w[:3,:3]=Rotation.from_rotvec([.3,.2,-.1]).as_matrix();w[:3,3]=[20,3,-7]
        world={k:w@v for k,v in pose.items()};r=world['pelvis'][:,:3,:3]
        vectors=np.stack([world[b][:,:3,3]-world[a][:,:3,3] for _,a,b in self.s.pairs],axis=1)
        local=np.einsum('nji,nkj->nki',r,vectors);local/=np.linalg.norm(local,axis=2)[:,:,None]
        np.testing.assert_allclose(local,self.s.evaluate(q),atol=2e-14,rtol=0)
    def test_named_reordering(self):
        names=self.t.names[::-1];np.testing.assert_array_equal(self.t.order(self.seed[:2,::-1],names),self.seed[:2])
    def test_bad_names(self):
        for names in (self.t.names[:-1],['head_yaw_joint']+self.t.names[1:],[self.t.names[1]]+self.t.names[1:]):
            with self.assertRaises(ValueError):self.t.order(self.seed[:1],names)
    def test_nonfinite_and_bad_shapes(self):
        for q in (np.zeros((1,28)),np.full((1,29),np.nan),np.zeros((998,29)),np.zeros(29)):
            with self.assertRaises(ValueError):self.t.evaluate(q)
    def test_wrong_hash_or_role(self):
        with self.assertRaises(ValueError):DirectionModel(SOURCE,'target')
        with self.assertRaises(ValueError):pairs('auto')
    def test_no_source_jacobian_claim(self):
        with self.assertRaises(ValueError):self.s.evaluate(np.zeros((1,29)),True)
    def test_head_excluded(self):self.assertEqual(len(self.t.names),29);self.assertFalse(any('head' in x for x in self.t.names))
    def test_direction_statistics(self):
        u=self.t.evaluate(self.seed[:2]);s=direction_statistics(u,u);self.assertEqual(s['mean_degrees'],0.)
        with self.assertRaises(ValueError):direction_statistics(u,np.zeros_like(u))
    def test_hard_rows_match_whole_sequence(self):
        first,last=100,102;g,h=joint_inequalities(self.t,self.seed,first,last);q=self.seed[first:last+1].copy();q[:,0]+=.001
        self.assertLessEqual(float(np.max(g@q.reshape(-1)-h)),1e-12)
        q[0,0]+=.9;self.assertGreater(float(np.max(g@q.reshape(-1)-h)),0)
    def test_global_endpoints(self):
        for a,b in ((0,2),(994,996),(0,0),(996,996)):
            g,h=joint_inequalities(self.t,self.seed,a,b);self.assertLessEqual(float(np.max(g@self.seed[a:b+1].reshape(-1)-h)),1e-12)
    def test_window_validation(self):
        for a,b in ((-1,1),(3,2),(0,93),(996,997),(0.,2),(True,2)):
            with self.assertRaises(ValueError):joint_inequalities(self.t,self.seed,a,b)
    def test_one_lp_synthetic_feasible_fit(self):
        seed=self.seed.copy();desired_q=seed[100:102].copy();desired_q[:,0]+=.01;desired_q[:,3]+=.01
        gate=Gate();q,r=solve(self.t,seed,self.t.evaluate(desired_q),100,101,gate,outer_limit=1)
        self.assertEqual(r['solver_calls'],1);self.assertEqual(gate.count,1);self.assertLess(r['final']['sum_l1_chordal'],r['initial']['sum_l1_chordal'])
        self.assertLessEqual(r['full_sequence_velocity_max_violation_rad_s'],1e-6);np.testing.assert_array_equal(seed,self.seed)
        for key in ('candidate_motion_generated','accepted_reference','training_ready','world_root_optimized','terrain_constraints_implemented'):self.assertIs(r[key],False)
    def test_bad_or_missing_targets(self):
        for desired in (None,np.zeros((2,8,3)),np.ones((2,8,2))):
            with self.assertRaises(ValueError):solve(self.t,self.seed,desired,100,101,Gate(),outer_limit=1)
    def test_finite_iteration_budget(self):
        with self.assertRaises(ValueError):solve(self.t,self.seed,None,100,101,Gate(),outer_limit=13)
    def test_first_wrist_not_rotated_by_any_wrist_dof(self):
        for path,kind in ((SOURCE,'source'),(TARGET,'target')):
            model=DirectionModel(path,kind,wrist_endpoint='first_wrist_pivot');q=np.zeros((2,29))
            for i,n in enumerate(model.names):
                if 'wrist' in n:q[1,i]=.4
            np.testing.assert_allclose(model.evaluate(q)[0,[3,7]],model.evaluate(q)[1,[3,7]],atol=1e-13,rtol=0)
    def test_first_wrist_target_jacobian(self):
        model=DirectionModel(TARGET,'target',wrist_endpoint='first_wrist_pivot');q=self.seed[:1].copy();u,j=model.evaluate(q,True)
        for col in range(29):
            d=np.zeros_like(q);d[:,col]=1e-6
            np.testing.assert_allclose((model.evaluate(q+d)-model.evaluate(q-d))/(2e-6),j[:,:,:,col],atol=2e-9,rtol=0)
    def test_endpoint_declaration_and_validation(self):
        self.assertEqual(pairs('source','first_wrist_pivot')[3][2],'left_wrist_roll_link')
        self.assertEqual(pairs('target','first_wrist_pivot')[3][2],'left_wrist_yaw_link')
        with self.assertRaises(ValueError):pairs('source','auto')
    def test_trust_radius_guard(self):
        for v in (0,.16,np.nan,True):
            with self.assertRaises(ValueError):solve(self.t,self.seed,None,100,101,Gate(),trust_radius=v)

if __name__=='__main__':unittest.main()