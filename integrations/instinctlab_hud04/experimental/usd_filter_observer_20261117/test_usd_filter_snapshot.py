import copy
import json
from pathlib import Path
import uuid
import unittest
from pxr import Sdf,Usd,UsdGeom,UsdPhysics
from usd_filter_snapshot import collect,compare_inventory,write_new


class SnapshotTests(unittest.TestCase):
    def make(self):
        stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/Robot')
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath('/Robot'))
        for name in ['A','B']:
            prim=UsdGeom.Xform.Define(stage,'/Robot/'+name).GetPrim();UsdPhysics.RigidBodyAPI.Apply(prim)
            collider=UsdGeom.Cube.Define(stage,'/Robot/'+name+'/shape').GetPrim();UsdPhysics.CollisionAPI.Apply(collider)
        joint=UsdPhysics.RevoluteJoint.Define(stage,'/Robot/J')
        joint.CreateBody0Rel().SetTargets(['/Robot/A']);joint.CreateBody1Rel().SetTargets(['/Robot/B'])
        return stage,joint

    def test_resolved_joint_fallback_is_false_not_authored(self):
        stage,_=self.make();r=collect(stage,'/Robot');a=r['joints'][0]['collision_enabled']
        self.assertEqual(a['value'],False);self.assertFalse(a['authored'])

    def test_explicit_joint_true_is_retained_without_exclusion(self):
        stage,j=self.make();j.CreateCollisionEnabledAttr(True);r=collect(stage,'/Robot')
        self.assertTrue(r['joints'][0]['collision_enabled']['value']);self.assertTrue(r['joints'][0]['collision_enabled']['authored'])
        self.assertEqual(r['automatic_pair_exclusions'],0)

    def test_explicit_joint_false_is_retained(self):
        stage,j=self.make();j.CreateCollisionEnabledAttr(False)
        self.assertTrue(collect(stage,'/Robot')['joints'][0]['collision_enabled']['authored'])

    def test_collider_fallback_true_is_distinct(self):
        stage,_=self.make();a=collect(stage,'/Robot')['colliders'][0]['collision_enabled']
        self.assertTrue(a['value']);self.assertFalse(a['authored'])

    def test_disabled_collider_is_not_omitted(self):
        stage,_=self.make();UsdPhysics.CollisionAPI(stage.GetPrimAtPath('/Robot/A/shape')).CreateCollisionEnabledAttr(False)
        r=collect(stage,'/Robot');self.assertEqual(len(r['colliders']),2);self.assertFalse(r['colliders'][0]['collision_enabled']['value'])

    def test_missing_physx_schema_remains_unknown(self):
        stage,_=self.make();a=collect(stage,'/Robot')['articulations'][0]['self_collision']
        self.assertFalse(a['present']);self.assertIsNone(a['value'])

    def test_plain_authored_physx_value_observed_not_validated(self):
        stage,_=self.make();stage.GetPrimAtPath('/Robot').CreateAttribute('physxArticulation:enabledSelfCollisions',Sdf.ValueTypeNames.Bool).Set(True)
        r=collect(stage,'/Robot');self.assertTrue(r['articulations'][0]['self_collision']['value']);self.assertFalse(r['actual_physx_pair_filters_verified'])

    def test_external_filter_owner_is_recorded(self):
        stage,_=self.make();outside=UsdGeom.Xform.Define(stage,'/External').GetPrim()
        UsdPhysics.FilteredPairsAPI.Apply(outside).CreateFilteredPairsRel().SetTargets(['/Robot/B'])
        r=collect(stage,'/Robot');self.assertEqual(r['global_filtered_pairs'],[dict(owner='/External',targets=['/Robot/B'])])

    def test_collision_group_collections_recorded_not_interpreted(self):
        stage,_=self.make();group=UsdPhysics.CollisionGroup.Define(stage,'/Groups/G')
        group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets(['/Robot/A'])
        row=collect(stage,'/Robot')['collision_groups'][0]
        self.assertIn('/Robot/A',row['relationships']['collection:colliders:includes'])

    def test_instance_proxies_are_traversed(self):
        stage,_=self.make();UsdGeom.Xform.Define(stage,'/Source');a=stage.GetPrimAtPath('/Robot/A')
        instance=UsdGeom.Xform.Define(stage,'/Robot/Instance').GetPrim();instance.GetReferences().AddInternalReference('/Robot/A');instance.SetInstanceable(True)
        rows=collect(stage,'/Robot')['colliders'];self.assertEqual(len(rows),3);self.assertTrue(any(r['instance_proxy'] for r in rows))

    def test_body_mapping_no_collision_index_guess(self):
        stage,_=self.make();r=collect(stage,'/Robot');expected=dict(shapes=[dict(link='A'),dict(link='B')])
        c=compare_inventory(r,expected);self.assertTrue(c['exact_body_leaf_counts_match']);self.assertFalse(c['matching_counts_prove_cooked_geometry'])

    def test_extra_shape_count_mismatch_stays_false(self):
        stage,_=self.make();self.assertFalse(compare_inventory(collect(stage,'/Robot'),dict(shapes=[dict(link='A')]))['exact_body_leaf_counts_match'])

    def test_static_shape_does_not_invent_body(self):
        stage,_=self.make();p=UsdGeom.Sphere.Define(stage,'/Robot/static').GetPrim();UsdPhysics.CollisionAPI.Apply(p)
        r=collect(stage,'/Robot');self.assertIsNone(r['colliders'][-1]['rigid_body_path'])

    def test_stage_units_are_not_source_motion_proof(self):
        stage,_=self.make();UsdGeom.SetStageMetersPerUnit(stage,1.0);r=collect(stage,'/Robot')
        self.assertEqual(r['meters_per_unit'],1);self.assertTrue(r['meters_per_unit_authored']);self.assertFalse(r['source_motion_units_or_alignment_proven'])

    def test_unset_units_marked_unauthored(self):
        stage,_=self.make();self.assertFalse(collect(stage,'/Robot')['meters_per_unit_authored'])

    def test_nonfinite_dimension_fails_closed(self):
        stage,_=self.make();UsdGeom.Cube(stage.GetPrimAtPath('/Robot/A/shape')).CreateSizeAttr(float('nan'))
        with self.assertRaises(ValueError):collect(stage,'/Robot')

    def test_prim_budget_fails_no_partial_result(self):
        stage,_=self.make()
        with self.assertRaises(RuntimeError):collect(stage,'/Robot',max_prims=1)

    def test_missing_explicit_root_rejected(self):
        stage,_=self.make()
        for root in ['/','/Missing','Robot']:
            with self.assertRaises(ValueError):collect(stage,root)

    def test_mesh_points_are_hashed_not_claimed_cooked(self):
        stage,_=self.make();m=UsdGeom.Mesh.Define(stage,'/Robot/A/mesh');m.CreatePointsAttr([(0,0,0),(1,0,0),(0,1,0)])
        m.CreateFaceVertexIndicesAttr([0,1,2]);m.CreateFaceVertexCountsAttr([3]);UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        row=next(r for r in collect(stage,'/Robot')['colliders'] if 'mesh' in r)
        self.assertEqual(row['mesh']['point_count'],3);self.assertTrue(row['mesh']['usd_points_are_not_physx_cooked_vertices'])
        with self.assertRaises(RuntimeError):collect(stage,'/Robot',max_points=2)

    def test_joint_body_relation_absence_not_inferred(self):
        stage,j=self.make();j.GetBody1Rel().ClearTargets(True)
        self.assertEqual(collect(stage,'/Robot')['joints'][0]['body1'],[])

    def test_existing_output_never_overwritten(self):
        directory=Path(__file__).resolve().parent/'test_outputs'/uuid.uuid4().hex
        directory.mkdir(parents=True);p=directory/'x.json';write_new(p,dict(a=1))
        with self.assertRaises(FileExistsError):write_new(p,dict(a=2))
        self.assertEqual(json.loads(p.read_text()),dict(a=1))

    def test_no_promoted_acceptance_flags(self):
        stage,_=self.make();r=collect(stage,'/Robot')
        for name in ['cooked_geometry_verified','actual_physx_pair_filters_verified','contact_schedule_accepted','gpu_started','accepted_reference','training_ready']:
            self.assertFalse(r[name])

    def test_duplicate_body_leaf_names_not_merged_as_identity(self):
        stage,_=self.make();r=collect(stage,'/Robot');r['colliders'][1]['rigid_body_path']='/Robot/Other/A'
        c=compare_inventory(r,dict(shapes=[dict(link='A'),dict(link='A')]))
        self.assertFalse(c['exact_body_leaf_counts_match']);self.assertIn('A',c['ambiguous_body_leaf_names'])

    def test_topology_budget_catches_many_indices_few_points(self):
        stage,_=self.make();m=UsdGeom.Mesh.Define(stage,'/Robot/A/mesh');m.CreatePointsAttr([(0,0,0)])
        m.CreateFaceVertexIndicesAttr([0]*15);m.CreateFaceVertexCountsAttr([3]*5);UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        with self.assertRaises(RuntimeError):collect(stage,'/Robot',max_points=1)

    def test_world_transform_retained_in_gf_convention(self):
        stage,_=self.make();UsdGeom.Xformable(stage.GetPrimAtPath('/Robot/A')).AddTranslateOp().Set((1,2,3))
        row=collect(stage,'/Robot')['colliders'][0]
        self.assertEqual(row['world_transform_gf_row_vector'][3],[1,2,3,1])

    def test_disabled_joint_still_reports_relation_and_flag(self):
        stage,j=self.make();j.CreateJointEnabledAttr(False);r=collect(stage,'/Robot')['joints'][0]
        self.assertFalse(r['joint_enabled']['value']);self.assertEqual(r['body0'],['/Robot/A'])


if __name__=='__main__':unittest.main(verbosity=2)