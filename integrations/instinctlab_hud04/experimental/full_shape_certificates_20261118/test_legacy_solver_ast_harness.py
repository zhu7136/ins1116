"""Run legacy synthetic regression assertions without lost private-workbench imports.

Loads actual repository function ASTs, never a model/data/optimizer constructor.
Unpatched heavy dependencies fail closed. This is NOT an integrated SDK test.
"""
import ast
import importlib.util
from pathlib import Path
import sys
import time
import types
import unittest
import warnings
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

BASE=Path(__file__).resolve().parents[1]


def no_heavy_work(*args,**kwargs):
    raise AssertionError('Unmocked heavy dependency in synthetic AST regression')


def functions(path,names,scope):
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('Source function missing')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),scope)


class LegacySolverAssertions(unittest.TestCase):
    def test_legacy_synthetic_assertions_with_real_function_bodies(self):
        solver=types.ModuleType('wholebody_solver');scope=solver.__dict__
        scope.update(np=np,sparse=sparse,time=time,warnings=warnings,OptimizeWarning=Warning,Rotation=Rotation)
        for name in ['hard_rows','joint_feasibility','temporal_system','temporal_values','linprog','WholeBodyTerms']:
            scope[name]=no_heavy_work
        functions(BASE/'world_candidate_20261118/world_features.py',['check_roots','perturb','weights'],scope)
        functions(BASE/'wholebody_guarded_candidates_20261118/wholebody_solver.py',['plan','geometry_cost','solve'],scope)
        # LP helpers are not evaluated by the old assertions except constructing
        # synthetic sparse rows; preserve their exact source as well.
        functions(BASE/'world_candidate_20261118/collision_temporal_solver.py',['l1_inequalities'],scope)
        previous=dict(sys.modules);old_path=sys.path.copy();result=unittest.TestResult()
        try:
            sys.modules['wholebody_solver']=solver
            sys.path.insert(0,str(BASE/'wholebody_guarded_candidates_20261118'))
            suite=unittest.TestSuite()
            for name in ['test_wholebody_solver','test_per_pose_foot_guard']:
                spec=importlib.util.spec_from_file_location(name,BASE/'wholebody_guarded_candidates_20261118'/f'{name}.py')
                module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
                suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
            suite.run(result)
        finally:
            sys.path[:]=old_path
            for name in ['wholebody_solver','test_wholebody_solver','test_per_pose_foot_guard','per_pose_foot_guard']:
                if name in previous:sys.modules[name]=previous[name]
                else:sys.modules.pop(name,None)
        self.assertEqual(result.testsRun,7)
        self.assertEqual(result.errors,[],msg=str(result.errors))
        self.assertEqual(result.failures,[],msg=str(result.failures))
        self.assertEqual(result.skipped,[])


if __name__=='__main__':unittest.main(verbosity=2)