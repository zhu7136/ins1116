import hashlib,os,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from named_target import NamedTarget,MODEL_SHA256,perturb
from window_problem import build,frame_edges,named_order,direction_residual,summarize

DELIVERY=Path(os.environ.get('HUD04_DELIVERY_ROOT','/workspace/hud04-recovery-bf4a057'))
MODEL=DELIVERY/'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
FIT=Path(os.environ.get('HUD04_JOINT_FIT','/workspace/hud04-implementation-20261117/step1.joint_fit.npz'))

class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t=NamedTarget(MODEL)
        if hashlib.sha256(FIT.read_bytes()).hexdigest()!='48f8d43e058420e5ff89861d2336ead3317dc18223c36a7305503f3df54a66eb':raise ValueError('Fit checksum')
        with np.load(FIT,allow_pickle=False) as z:
            cls.q=z['joint_pos'].astype(float);cls.names=z['joint_names'].tolist()
    def roots(self,n):
        out=np.tile(np.eye(4),(n,1,1));out[:,:3,3]=[2.,-1.,1.];out[:,:3,:3]=Rotation.from_rotvec([.2,-.1,.3]).as_matrix();return out
    def system(self):return build(self.t,self.q[[611,612,659]],self.names,self.roots(3),[611,612,659])
    def test_shapes_and_no_head(self):
        s=self.system();self.assertEqual(s.tracking_jacobian.shape,(117,105));self.assertEqual(s.hard_matrix.shape,(116,105))
        self.assertFalse(any('head' in n for n in s.joint_names))
    def test_no_silent_targets(self):
        s=self.system();self.assertIsNone(s.tracking_residual);self.assertFalse(s.target_objective_supplied)
    def test_explicit_targets_actual_residual(self):
        q=self.q[[611,612]];roots=self.roots(2)
        p=np.array([self.t.evaluate(row[named_order(self.names,self.t.names)],root,False) for row,root in zip(q,roots)])
        s=build(self.t,q,self.names,roots,[611,612],desired_points=p+.1)
        np.testing.assert_allclose(s.tracking_residual,-.1,atol=1e-15);self.assertTrue(s.target_objective_supplied)
    def test_no_cross_gap_velocity_rows(self):self.assertEqual(self.system().edges,[(0,1)])
    def test_joint_rows_match_direct_model_residual(self):
        s=self.system();rng=np.random.default_rng(32);d=rng.normal(size=(3,35))*.001
        q=self.q[[611,612,659]][:,named_order(self.names,self.t.names)];new=q+d[:,6:]
        value=s.hard_matrix@d.ravel()
        np.testing.assert_allclose((s.hard_upper-value)[:87].reshape(3,29),self.t.upper-new,atol=1e-15)
        np.testing.assert_allclose((value-s.hard_lower)[:87].reshape(3,29),new-self.t.lower,atol=1e-15)
        np.testing.assert_allclose((s.hard_upper-value)[87:],self.t.vmax/90-(new[1]-new[0]),atol=1e-15)
    def test_root_columns_do_not_enter_hard_joint_limits(self):
        s=self.system();d=np.zeros((3,35));d[:,:6]=.01
        np.testing.assert_array_equal(s.hard_matrix@d.ravel(),0)
    def test_seed_remains_joint_feasible(self):self.assertLessEqual(summarize(self.system())['zero_joint_increment_max_violation'],1e-6)
    def test_named_order_reversal(self):
        a=self.system();b=build(self.t,self.q[[611,612,659]][:,::-1],self.names[::-1],self.roots(3),[611,612,659])
        np.testing.assert_array_equal(a.hard_lower,b.hard_lower);np.testing.assert_array_equal(a.tracking_jacobian.toarray(),b.tracking_jacobian.toarray())
    def test_reject_missing_duplicate_head_names(self):
        for names in [self.names[:-1],self.names[:-1]+[self.names[0]],self.names[:-1]+['head_yaw_joint']]:
            with self.assertRaises(ValueError):named_order(names,self.t.names)
    def test_invalid_frames(self):
        for frames in [[0,0],[2,1],np.array([2,1],dtype=np.uint64),[-1],[997],[1.0],[True],[]]:
            with self.assertRaises(ValueError):frame_edges(frames)
    def test_no_resampling_fps(self):
        for fps in [50,90.,True]:
            with self.assertRaises(ValueError):build(self.t,self.q[:2],self.names,self.roots(2),[0,1],fps=fps)
    def test_bounded_window_size(self):
        with self.assertRaises(ValueError):build(self.t,self.q[:94],self.names,self.roots(94),np.arange(94))
    def test_no_input_mutation(self):
        q=self.q[:2].copy();root=self.roots(2);old=q.copy();oldroot=root.copy();build(self.t,q,self.names,root,[0,1])
        np.testing.assert_array_equal(q,old);np.testing.assert_array_equal(root,oldroot)
    def test_invalid_trust_region(self):
        for v in [0,-.01,.021,float('nan'),True]:
            with self.assertRaises(ValueError):build(self.t,self.q[:2],self.names,self.roots(2),[0,1],root_translation_step=v)
    def test_reflection_root_rejected(self):
        root=self.roots(2);root[0,:3,:3]=np.diag([-1,1,1])
        with self.assertRaises(ValueError):build(self.t,self.q[:2],self.names,root,[0,1])
    def test_bad_target_shapes_rejected(self):
        for desired in [np.zeros((2,1,3)),np.full((2,13,3),float('nan'))]:
            with self.assertRaises(ValueError):build(self.t,self.q[:2],self.names,self.roots(2),[0,1],desired_points=desired)
    def test_jacobian_all_columns_finite_difference(self):
        q=self.q[611,named_order(self.names,self.t.names)];root=self.roots(1)[0];p,j=self.t.evaluate(q,root);eps=1e-6
        for c in range(35):
            d=np.zeros(35);d[c]=eps
            plus=self.t.evaluate(q+d[6:],perturb(root,d[:6]),False);minus=self.t.evaluate(q-d[6:],perturb(root,-d[:6]),False)
            np.testing.assert_allclose((plus-minus)/(2*eps),j[:,:,c],atol=2e-8,rtol=0)
    def test_direction_jacobian_finite_difference(self):
        q=self.q[611,named_order(self.names,self.t.names)];root=self.roots(1)[0];p,j=self.t.evaluate(q,root)
        desired=np.array([0.,0.,1.]);r,jd=direction_residual(p,j,1,2,desired);eps=1e-6
        for c in range(35):
            d=np.zeros(35);d[c]=eps
            plus=self.t.evaluate(q+d[6:],perturb(root,d[:6]),False);minus=self.t.evaluate(q-d[6:],perturb(root,-d[:6]),False)
            u=lambda a:(a[2]-a[1])/np.linalg.norm(a[2]-a[1])
            np.testing.assert_allclose((u(plus)-u(minus))/(2*eps),jd[:,c],atol=2e-8,rtol=0)
        np.testing.assert_allclose(jd[:,:3],0,atol=1e-15)
    def test_direction_translation_and_scale_invariance(self):
        q=self.q[611,named_order(self.names,self.t.names)];p,j=self.t.evaluate(q,self.roots(1)[0])
        r,a=direction_residual(p,j,1,2,[0,0,1]);r2,a2=direction_residual(3*p+7,3*j,1,2,[0,0,1])
        np.testing.assert_allclose(r,r2,atol=1e-14);np.testing.assert_allclose(a,a2,atol=1e-14)
    def test_direction_rejects_degenerate_or_nonunit(self):
        p=np.zeros((2,3));j=np.zeros((2,3,35))
        for desired in [[0,0,0],[0,0,2],[0,0,1]]:
            with self.assertRaises(ValueError):direction_residual(p,j,0,1,desired)
    def test_root_rotation_about_root(self):
        root=self.roots(1)[0];moved=perturb(root,[0,0,0,.1,0,0]);np.testing.assert_array_equal(moved[:3,3],root[:3,3])
    def test_conservative_summary(self):
        r=summarize(self.system())
        for k in ['accepted_reference','training_ready','solver_implemented','terrain_constraints_implemented','self_collision_constraints_implemented','candidate_motion_generated']:self.assertIs(r[k],False)
    def test_all_sources_and_patched_runner_compile(self):
        import ast
        root=Path(__file__).parent
        # The reviewed patch was published in the actual implementation folder,
        # not the old sandbox-only patches/ staging directory. Compile/inspect
        # the real delivered runner; never execute its solver entrypoint.
        runner=root.parent/'implementation_20261117/run_joint_fit.py'
        for path in [*root.glob('*.py'),runner]:compile(path.read_text(),str(path),'exec')
        tree=ast.parse(runner.read_text())
        worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        archive_call=next(n for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='freeze_file')
        sends=[n for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='send']
        self.assertTrue(sends);self.assertTrue(all(archive_call.lineno<n.lineno for n in sends))

if __name__=='__main__':unittest.main(verbosity=2)