"""Endpoint policy and real-solver AST tests; fake LP, no motion optimization."""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
import warnings
import numpy as np
from scipy import sparse
from wholebody_step_guard import EndpointState, compare_states, WholeBodyStepGuard


def state(strict=(), unresolved=()):
    return EndpointState('a'*64,(665,),frozenset(strict),frozenset(unresolved),903,43*8068)


SELF=('self',665,1,2)
TERRAIN=('terrain',665,1,30)


class PolicyTests(unittest.TestCase):
    def test_preserved_separation(self):
        self.assertTrue(compare_states(state(),state())[0])

    def test_new_strict_self_rejected(self):
        self.assertFalse(compare_states(state(),state([SELF]))[0])

    def test_new_strict_terrain_rejected(self):
        self.assertFalse(compare_states(state(),state([TERRAIN]))[0])

    def test_new_unresolved_is_not_clear(self):
        self.assertFalse(compare_states(state(),state(unresolved=[TERRAIN]))[0])

    def test_unresolved_to_strict_rejected(self):
        self.assertFalse(compare_states(state(unresolved=[SELF]),state([SELF]))[0])

    def test_existing_strict_not_accepted_reference(self):
        passed,detail=compare_states(state([SELF]),state([SELF]))
        self.assertTrue(passed);self.assertFalse(detail['accepted_reference'])
        self.assertFalse(detail['penetration_depth_comparison'])

    def test_same_count_new_pair_rejected(self):
        self.assertFalse(compare_states(state([SELF]),state([TERRAIN]))[0])

    def test_resolving_ambiguity_can_pass(self):
        self.assertTrue(compare_states(state(unresolved=[SELF]),state())[0])

    def test_frame_binding_mismatch(self):
        with self.assertRaises(ValueError):compare_states(state(),replace(state(),frames=(666,)))

    def test_geometry_binding_mismatch(self):
        with self.assertRaises(ValueError):compare_states(state(),replace(state(),binding='b'*64))

    def test_partial_accounting_rejected(self):
        with self.assertRaises(ValueError):compare_states(state(),replace(state(),self_records=902))

    def test_contradictory_classification_rejected(self):
        with self.assertRaises(ValueError):compare_states(state([SELF],[SELF]),state())

    def test_outside_frame_rejected(self):
        with self.assertRaises(ValueError):compare_states(state([('self',3,1,2)]),state())

    def test_unknown_inventory_rejected(self):
        with self.assertRaises(ValueError):compare_states(state([('terrain',665,43,0)]),state())

    def test_unknown_class_rejected(self):
        with self.assertRaises(ValueError):compare_states(state([('ignored',665,1,2)]),state())

    def test_empty_frames_rejected(self):
        with self.assertRaises(ValueError):compare_states(replace(state(),frames=()),replace(state(),frames=()))

    def test_narrowphase_partial_masks_rejected(self):
        with self.assertRaises(ArithmeticError):
            WholeBodyStepGuard._record(dict(separated=[False],strict=[False],unresolved=[False]),np.array([0]),np.array([665]),'self',1,2,set(),set())

    def test_narrowphase_identity_records(self):
        hit=set();unknown=set()
        WholeBodyStepGuard._record(dict(separated=[False,False],strict=[True,False],unresolved=[False,True]),np.array([0,0]),np.array([665]),'terrain',1,np.array([30,31]),hit,unknown)
        self.assertEqual(hit,{TERRAIN});self.assertEqual(unknown,{('terrain',665,1,31)})


class FakeModel:
    def evaluate(self,q,roots,derivatives=False):
        value=np.zeros((len(q),90));value[:,0]=q[:,0]
        if not derivatives:return value
        jac=np.zeros((len(q),90,35));jac[:,0,6]=1
        return value,jac


class FakeBody:
    margin=.001
    def evaluate(self,model,q,roots,jacobian=False):
        gap=np.array([1.]);records=[dict(kind='terrain',frame=0,first=0,second=0)]
        return (gap,sparse.csr_matrix((1,len(q)*35)),records,{}) if jacobian else (gap,records,{})


class StubGuard:
    def __init__(self):self.scans=[]
    def scan(self,model,q,roots,frames):
        self.scans.append(float(q[0,0]))
        return state([TERRAIN]) if q[0,0]>.075 else state()
    compare=staticmethod(compare_states)


class WiringTests(unittest.TestCase):
    def solver(self):
        path=Path(__file__).resolve().parents[1]/'wholebody_guarded_candidates_20261118/wholebody_solver.py'
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='solve')
        calls=[]
        def fake_lp(c,**kwargs):
            calls.append(1);x=np.zeros(len(c));x[6]=.1
            return SimpleNamespace(success=True,status=0,nit=0,x=x)
        def inequalities(a,res,nv):
            return sparse.vstack([sparse.hstack([a,-sparse.eye(len(res))]),sparse.hstack([-a,-sparse.eye(len(res))])]),np.r_[-res,res]
        scope=dict(np=np,sparse=sparse,time=time,warnings=warnings,OptimizeWarning=Warning,
            plan=lambda:[(665,665)],weights=lambda:np.ones(90),check_roots=lambda r,n:r,
            joint_feasibility=lambda m,q:dict(position_violation=0.,velocity_violation=0.),
            WholeBodyTerms=FakeBody,geometry_cost=lambda body,gaps:0.,
            hard_rows=lambda m,q,f,l:(sparse.csr_matrix((1,35)),np.ones(1)),
            temporal_system=lambda q,r,f,l:(np.zeros(1),sparse.csr_matrix((1,35))),
            temporal_values=lambda q,r,f,l:np.zeros(1),l1_inequalities=inequalities,
            linprog=fake_lp,perturb=lambda roots,delta:roots.copy())
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
        return scope['solve'],calls

    def test_merit_improving_collision_rejected_before_assignment(self):
        solve,calls=self.solver();guard=StubGuard();rows=[]
        q=np.zeros((997,29));r=np.repeat(np.eye(4)[None],997,axis=0);desired=np.zeros((997,90));desired[:,0]=1
        result,_,report=solve(FakeModel(),q,r,desired,rows.append,certificate_guard=guard)
        self.assertEqual(calls,[1]);self.assertEqual(rows[0]['alpha'],.5)
        self.assertFalse(rows[0]['backtracking'][0]['endpoint_certificate_guard_passed'])
        self.assertTrue(rows[0]['backtracking'][1]['endpoint_certificate_guard_passed'])
        self.assertEqual(guard.scans,[0.,.1,.05]);self.assertAlmostEqual(result[665,0],.05)
        np.testing.assert_array_equal(q,0)
        self.assertTrue(report['all43_all903_endpoint_certificate_guard_required'])
        self.assertFalse(report['accepted_reference'])

    def test_missing_guard_refuses_before_lp_or_motion_changes(self):
        solve,calls=self.solver()
        with self.assertRaisesRegex(ValueError,'guard required'):
            solve(None,None,None,None,None)
        self.assertFalse(calls)


if __name__=='__main__':unittest.main(verbosity=2)