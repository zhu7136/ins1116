"""CPU fake-object regressions for task301 metadata gaps, NOT runtime proof."""
import ast
from pathlib import Path
from types import SimpleNamespace as S
import unittest

from prepare_runtime_observation import (
    configured_scene_path,existing_stage_snapshot,prim_metadata,scalar_or_small_array,
)
from native_contact_observer import exact_path


class FakeQuat:
    def __init__(self,w,xyz):self.w=w;self.xyz=xyz
    def GetReal(self):return self.w
    def GetImaginary(self):return self.xyz


class MetadataRevisionTests(unittest.TestCase):
    def test_quaternion_explicit_wxyz_not_normalized(self):
        r=scalar_or_small_array(FakeQuat(2.,(3.,4.,5.)),quaternion_types=(FakeQuat,))
        self.assertEqual(r['order'],'wxyz')
        self.assertEqual(r['components'],[2.,3.,4.,5.])
        self.assertFalse(r['normalized_by_collector'])

    def test_nested_quaternion_retains_registration(self):
        r=scalar_or_small_array([[FakeQuat(1.,(0.,0.,0.))]],quaternion_types=(FakeQuat,))
        self.assertEqual(r[0][0]['components'],[1.,0.,0.,0.])

    def test_unregistered_protocol_not_called(self):
        calls=[]
        value=S(GetReal=lambda:calls.append('forbidden'),GetImaginary=lambda:calls.append('forbidden'))
        with self.assertRaises(TypeError):scalar_or_small_array(value)
        self.assertEqual(calls,[])

    def test_quaternion_invalid_components_and_budgets_fail(self):
        for q in (FakeQuat(float('nan'),(0.,0.,0.)),FakeQuat(1.,(0.,float('inf'),0.)),
                  FakeQuat(1.,(0.,0.)),FakeQuat(1.,range(5))):
            with self.subTest(q=q):
                with self.assertRaises(ValueError):scalar_or_small_array(q,quaternion_types=(FakeQuat,))
        with self.assertRaises(ValueError):
            scalar_or_small_array(FakeQuat(1.,(0.,0.,0.)),limit=3,quaternion_types=(FakeQuat,))

    def test_prim_retains_authored_default_time_semantics(self):
        attr=S(GetName=lambda:'physics:principalAxes',Get=lambda:FakeQuat(1.,(0.,0.,0.)),
               HasAuthoredValueOpinion=lambda:True,GetNumTimeSamples=lambda:0)
        prim=S(IsValid=lambda:True,GetAttributes=lambda:[attr],GetRelationships=lambda:[],
               GetTypeName=lambda:'Xform',IsInstanceProxy=lambda:False)
        v=prim_metadata(prim,quaternion_types=(FakeQuat,))['attributes']['physics:principalAxes']['value']
        self.assertTrue(v['authored'])
        self.assertEqual(v['time_code'],'USD_DEFAULT_NOT_CURRENT_PHYSICS_STATE')
        self.assertEqual(v['value']['components'],[1.,0.,0.,0.])

    def test_root_scene_metadata_path_read_exactly_without_fallback(self):
        path=configured_scene_path(S(physics_prim_path='/physicsScene'));calls=[]
        def lookup(p):calls.append(p);return None
        r=existing_stage_snapshot(S(GetPrimAtPath=lookup),S(get=lambda p:None),S(),paths=[path])
        self.assertEqual(calls,['/physicsScene'])
        self.assertEqual(r['prims'][path]['status'],'UNKNOWN_OR_ERROR')
        self.assertFalse(r['final_solver_filter_verified'])

    def test_missing_or_invalid_scene_config_not_guessed(self):
        with self.assertRaises(AttributeError):configured_scene_path(S())
        for path in (None,'','/','/physicsScene/','/a//b','/a/../b','/a.b','/a*','relative','/scene\n'):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):configured_scene_path(S(physics_prim_path=path))

    def test_metadata_invalid_path_rejected_before_any_stage_read(self):
        calls=[];stage=S(GetPrimAtPath=lambda p:calls.append(p))
        for paths in (['/a','/b*'],['/a']*2,['/scene'+str(i) for i in range(33)]):
            with self.assertRaises(ValueError):existing_stage_snapshot(stage,None,None,paths=paths)
        self.assertEqual(calls,[])

    def test_actor_registry_restriction_not_broadened(self):
        self.assertEqual(exact_path('/World/a'),'/World/a')
        for path in ('/physicsScene','/Looks/material'):
            with self.assertRaises(ValueError):exact_path(path)

    def test_host_adapter_uses_explicit_types_and_config_no_old_path(self):
        text=Path(__file__).with_name('runtime_trial.py').read_text();tree=ast.parse(text)
        self.assertNotIn('/World/physicsScene',text)
        self.assertTrue(any(isinstance(n,ast.Call) and ast.unparse(n.func)=='configured_scene_path'
                            and ast.unparse(n.args[0])=='sim.cfg' for n in ast.walk(tree)))
        self.assertIn('quaternion_types=(Gf.Quatd, Gf.Quatf, Gf.Quath)',text)


if __name__=='__main__':unittest.main()