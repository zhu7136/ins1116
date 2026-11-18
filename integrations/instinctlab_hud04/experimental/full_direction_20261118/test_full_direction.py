"""CPU software checks; synthetic fixture solves are NOT source-motion trials."""
import unittest
import numpy as np
from inputs import load,dependency
from full_direction_solver import inequalities,validate,solve,stats

def spawn_fixture(gate,payload):
    gate.step()
    return {'spawn_path_import_verified':True,'payload':payload}

class Gate:
    def __init__(self):self.n=0
    def step(self):self.n+=1

class FullTests(unittest.TestCase):
    @classmethod
    def setUpClass(c):
        c.s,c.t,c.seed,c.desired,c.binding=load()
        c.q=np.tile((c.t.lower+c.t.upper)/2,(3,1));c.u=c.t.evaluate(c.q)
    def test_997_input_binding(self):
        self.assertEqual(self.seed.shape,(997,29));self.assertEqual(self.desired.shape,(997,8,3));self.assertFalse(self.binding['source_root_fields_read'])
    def test_all_996_temporal_edges(self):
        g,h=inequalities(self.t,997);self.assertEqual(g.shape,(115594,28913));self.assertEqual(len(h),115594)
        self.assertLessEqual(float(np.max(g@self.seed.reshape(-1)-h)),1e-6)
    def test_direct_difference_rows(self):
        g,h=inequalities(self.t,3);q=self.q.copy();q[:,0]+=[.001,.003,-.002]
        np.testing.assert_allclose((g@q.reshape(-1))[174:232],np.diff(q,axis=0).reshape(-1),atol=1e-14)
    def test_old_window_seam_is_not_omitted(self):
        g,h=inequalities(self.t,997);q=np.tile((self.t.lower+self.t.upper)/2,(997,1));q[30,0]+=1.
        positive=(g@q.reshape(-1)-h)[2*997*29:3*997*29-29].reshape(996,29)
        self.assertGreater(positive[29,0],0.)
    def test_endpoints_are_free_not_anchored_to_seed(self):
        g,h=inequalities(self.t,3);q=self.q.copy()+.0001;self.assertLessEqual(float(np.max(g@q.reshape(-1)-h)),1e-12)
    def test_shape_and_budget_guards(self):
        for n in (0,1,998,3.,True):
            with self.assertRaises(ValueError):inequalities(self.t,n)
        for value in (0,13,True,1.):
            with self.assertRaises(ValueError):solve(self.t,self.q,self.u,Gate(),outer_limit=value)
    def test_input_nonfinite_and_missing_directions(self):
        for desired in (None,np.zeros((3,8,3)),np.full((3,8,3),np.nan),np.ones((2,8,3))):
            with self.assertRaises(ValueError):validate(self.t,self.q,desired)
        q=self.q.copy();q[0,0]=np.nan
        with self.assertRaises(ValueError):validate(self.t,q,self.u)
    def test_infeasible_seed_not_silently_clipped(self):
        q=self.q.copy();q[0,0]=self.t.upper[0]+.01
        with self.assertRaises(ValueError):validate(self.t,q,self.u)
        q=self.q.copy();q[0,0]+=1.
        with self.assertRaises(ValueError):validate(self.t,q,self.u)
    def test_lp_and_trust_caps(self):
        for name,values in (('lp_seconds',[0,4.1,np.nan,True]),('trust_radius',[0,.151,np.nan,True])):
            for value in values:
                with self.assertRaises(ValueError):solve(self.t,self.q,self.u,Gate(),**{name:value})
    def test_target_endpoint_required(self):
        original=self.t.wrist_endpoint
        try:
            self.t.wrist_endpoint='distal_chain'
            with self.assertRaises(ValueError):validate(self.t,self.q,self.u)
        finally:self.t.wrist_endpoint=original
    def test_unknown_lp_algorithm_rejected(self):
        for name in ('simplex','',None):
            with self.assertRaises(ValueError):solve(self.t,self.q,self.u,Gate(),lp_method=name)
    def test_ipm_small_fixture(self):
        q=self.q.copy();goal=q.copy();goal[:,0]+=.01
        out,r=solve(self.t,q,self.t.evaluate(goal),Gate(),outer_limit=1,lp_method='highs-ipm')
        self.assertEqual(r['lp_method'],'highs-ipm');self.assertEqual(r['accepted_steps'],1)
        self.assertLess(r['final']['sum_l1_chordal'],r['initial']['sum_l1_chordal'])
    def test_exact_target_has_zero_angle(self):self.assertEqual(stats(self.u,self.u)['mean_degrees'],0.)
    def test_small_synthetic_joint_fit(self):
        q=self.q.copy();desired_q=q.copy();desired_q[:,0]+=.01;desired_q[:,3]+=.01;gate=Gate();rows=[]
        out,r=solve(self.t,q,self.t.evaluate(desired_q),gate,outer_limit=1,progress=rows.append)
        self.assertEqual(gate.n,1);self.assertEqual(r['actual_lp_calls'],1);self.assertEqual(r['native_velocity_edges'],2)
        self.assertLess(r['final']['sum_l1_chordal'],r['initial']['sum_l1_chordal']);self.assertEqual(len(rows),1)
        np.testing.assert_array_equal(q,self.q)
        for key in ('candidate_motion_generated','accepted_reference','training_ready','world_root_optimized','terrain_constraints_implemented'):self.assertIs(r[key],False)
    def test_no_descent_does_not_claim_convergence(self):
        q,r=solve(self.t,self.q,self.u,Gate(),outer_limit=1)
        self.assertEqual(r['accepted_steps'],0);self.assertEqual(r['termination'],'no_nonlinear_descent_in_fixed_line_search')
        self.assertFalse(r['full_length_joint_diagnostic_generated']);np.testing.assert_array_equal(q,self.q)
    def test_pinned_dynamic_supervisor_is_spawn_importable(self):
        runtime=dependency('bounded_runtime.py')
        r=runtime.supervise('test_full_direction','spawn_fixture',[{'fixture':True}],runtime.Budget(window_seconds=5,total_seconds=5,iterations=1,numeric_threads=2,max_windows=1))
        self.assertEqual(r['status'],'completed');self.assertTrue(r['windows'][0]['child_reaped'])
        self.assertTrue(r['windows'][0]['diagnostic']['spawn_path_import_verified'])

if __name__=='__main__':unittest.main()