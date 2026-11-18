"""Synthetic solver-STUB control-flow tests; NO optimizer imported or called.

Uses the real experimental solve() AST and real FootTerms geometry methods.
Synthetic in-memory arrays are NOT candidate robot motions or results.
"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
import numpy as np
from scipy import sparse
from box_triangle_guard import BoxTriangleGuard

WORLD=Path(__file__).resolve().parents[1]/'world_candidate_20261118'
spec=importlib.util.spec_from_file_location('footterms_for_test',WORLD/'terrain_temporal_terms.py')
Terms=importlib.util.module_from_spec(spec);spec.loader.exec_module(Terms)


def foot_fixture():
    foot=Terms.FootTerms.__new__(Terms.FootTerms)
    foot.fresh_module=Terms.load_fresh_guard();foot.center=np.zeros(3);foot.halfsize=np.full(3,.04)
    foot.fresh_guard=BoxTriangleGuard(np.array([[[.1,-2,-2],[.1,2,-2],[.1,0,2]]]),tolerance=1e-9)
    foot.cuts=lambda model,q,roots:(np.ones(1),sparse.csr_matrix((1,len(q)*35)),[])
    foot.fixed_cut_gaps=lambda model,q,roots,records:np.ones(1)
    return foot


class FakeKinematics:
    def fk(self,q):
        poses=np.repeat(np.eye(4)[None],len(q),axis=0);poses[:,0,3]=q[:,0]
        return {s+'_ankle_roll_link':poses.copy() for s in ['left','right']},{}


class FakeModel:
    model=FakeKinematics()
    def evaluate(self,q,roots,derivatives=False):
        value=np.zeros((len(q),90));value[:,0]=q[:,0]
        if not derivatives:return value
        jac=np.zeros((len(q),90,35));jac[:,0,6]=1
        return value,jac


class WiringTests(unittest.TestCase):
    def test_both_real_foot_methods_detect_new_collision(self):
        foot=foot_fixture();model=FakeModel();q=np.zeros((2,29));r=np.repeat(np.eye(4)[None],2,axis=0)
        old=foot.fresh_geometry(model,q,r,frame_indices=np.array([611,612]));q[1,0]=.1
        new=foot.fresh_geometry(model,q,r,frame_indices=np.array([611,612]));decision=foot.reject_geometry_step(old,new)
        self.assertTrue(decision['reject'])
        self.assertEqual(decision['sides']['left']['new_hit_pose_indices'],[1])
        self.assertEqual(new['left']['pose_ids'],['left:611','left:612'])

    def test_real_solve_function_rejects_improved_merit_before_assignment(self):
        # No import/execution of collision_temporal_solver module or linprog.
        node=next(n for n in ast.parse((WORLD/'collision_temporal_solver.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='solve')
        foot=foot_fixture();stub_calls=[];rows=[]
        def stub_lp(c,**kwargs):
            stub_calls.append(1);x=np.zeros(len(c));x[:997*35].reshape(997,35)[:,6]=.1
            return SimpleNamespace(x=x,success=True,status=0,nit=0)
        scope=dict(np=np,sparse=sparse,time=time,model=FakeModel(),plan=lambda:[(0,996)],weights=lambda:np.ones(90),
            check_roots=lambda roots,n:roots,joint_feasibility=lambda model,q:dict(position_violation=0.,velocity_violation=0.),
            hard_rows=lambda model,q,first,last:(sparse.csr_matrix((1,(last-first+1)*35)),np.ones(1)),
            temporal_system=lambda q,r,first,last:(np.zeros(1),sparse.csr_matrix((1,(last-first+1)*35))),
            temporal_values=lambda q,r,first,last:np.zeros(1),FootTerms=lambda:foot,linprog=stub_lp,
            l1_inequalities=lambda a,res,nv:(sparse.vstack([sparse.hstack([a,-sparse.eye(len(res))]),sparse.hstack([-a,-sparse.eye(len(res))])]),np.r_[-res,res]),
            perturb=lambda roots,delta:roots.copy())
        import warnings
        scope.update(warnings=warnings,OptimizeWarning=Warning)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(WORLD/'collision_temporal_solver.py'),'exec'),scope)
        q=np.zeros((997,29));r=np.repeat(np.eye(4)[None],997,axis=0);desired=np.zeros((997,90));desired[:,0]=1
        result_q,_,report=scope['solve'](FakeModel(),q,r,desired,rows.append,max_calls=100)
        self.assertEqual(len(stub_calls),1);self.assertEqual(rows[0]['alpha'],.5)
        self.assertEqual(len(rows[0]['fresh_finite_triangle_geometry_rejections']),1)
        self.assertTrue(rows[0]['fresh_geometry_endpoint_guard_passed'])
        np.testing.assert_allclose(result_q[:,0],.05);np.testing.assert_array_equal(q,0)
        self.assertEqual(report['fresh_geometry_rejected_line_search_steps'],1)

    def test_frame_indices_required(self):
        foot=foot_fixture()
        with self.assertRaises(ValueError):foot.fresh_geometry(FakeModel(),np.zeros((2,29)),np.repeat(np.eye(4)[None],2,axis=0),frame_indices=np.array([1,1]))


if __name__=='__main__':unittest.main(verbosity=2)