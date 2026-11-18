import ast
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from material_evidence import audit_source, read_existing_articulation_materials, source_record

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
UPSTREAM = Path('/workspace/InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py')
CFG = REPO / 'integrations/instinctlab_hud04/overlay/instinctlab/tasks/shadowing/perceptive/config/hud04'
PATHS = [UPSTREAM, CFG/'perceptive_shadowing_cfg.py', CFG/'terrain_shadowing_cfg.py',
         REPO/'flux/hud04_asset_camera_probe/asset_config.py']


class View:
    count = 1
    max_shapes = 2

    def __init__(self, values=None):
        self.values = np.array([[[1., .8, 0.], [.4, .3, .1]]]) if values is None else values
        self.calls = 0

    def get_material_properties(self):
        self.calls += 1
        return self.values


class MaterialTests(unittest.TestCase):
    def test_real_source_declarations_not_effective_values(self):
        report = audit_source(*PATHS)
        self.assertEqual(report['terrain_material_explicit_keywords']['static_friction'], 1.)
        self.assertEqual(report['terrain_material_explicit_keywords']['dynamic_friction'], 1.)
        self.assertEqual(report['terrain_material_explicit_keywords']['friction_combine_mode'], 'multiply')
        self.assertTrue(report['hud04_startup_material_randomization_disabled'])
        self.assertFalse(report['robot_importer_explicit_physics_material_keyword'])
        self.assertIsNone(report['effective_pair_static_friction'])
        self.assertFalse(report['runtime_material_binding_verified'])

    def mutate_source(self, index, change):
        records = [source_record(p) for p in PATHS]
        records[index]['tree'] = ast.parse(change(ast.unparse(records[index]['tree'])))
        with patch('material_evidence.source_record', side_effect=records):
            return audit_source(*PATHS)

    def test_randomization_not_disabled_rejected(self):
        with self.assertRaisesRegex(ValueError, 'disable not established'):
            self.mutate_source(1, lambda s: s.replace("'physics_material',", "'different_term',"))

    def test_changed_terrain_restore_rejected(self):
        with self.assertRaisesRegex(ValueError, 'copy path changed'):
            self.mutate_source(2, lambda s: s.replace('PerceptiveShadowingSceneCfg().terrain.copy()', 'other_terrain()'))

    def test_missing_explicit_friction_rejected(self):
        with self.assertRaisesRegex(ValueError, 'friction declarations missing'):
            self.mutate_source(0, lambda s: s.replace('static_friction=1.0,', ''))

    def test_exact_order_and_one_getter(self):
        view = View()
        report = read_existing_articulation_materials(view, context_tag='CPU_FAKE_ONLY')
        self.assertEqual(view.calls, 1)
        self.assertEqual(report['column_order'], ['static_friction', 'dynamic_friction', 'restitution'])
        self.assertEqual(report['raw_material_slots_including_unresolved_padding'][0][1], [.4, .3, .1])
        self.assertIsNone(report['slot_to_collider_mapping'])
        self.assertIsNone(report['effective_pair_friction'])
        self.assertFalse(report['runtime_identity_verified'])
        self.assertFalse(report['training_ready'])

    def test_snapshot_not_alias(self):
        view = View()
        report = read_existing_articulation_materials(view, context_tag='CPU_FAKE_ONLY')
        view.values[:] = 99.
        self.assertEqual(report['raw_material_slots_including_unresolved_padding'][0][0][0], 1.)

    def test_budget_rejected_before_getter(self):
        for count, shapes in [(9, 2), (1, 257), (True, 2), (1.0, 2), (0, 2)]:
            view = View()
            view.count, view.max_shapes = count, shapes
            with self.assertRaises(ValueError):
                read_existing_articulation_materials(view, context_tag='CPU_FAKE_ONLY')
            self.assertEqual(view.calls, 0)

    def test_flat_shape_not_silently_reshaped(self):
        with self.assertRaisesRegex(ValueError, 'shape/type'):
            read_existing_articulation_materials(View(np.zeros(6)), context_tag='CPU_FAKE_ONLY')

    def test_nonfinite_rejected(self):
        view = View()
        view.values[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            read_existing_articulation_materials(view, context_tag='CPU_FAKE_ONLY')

    def test_padding_not_removed_or_bound(self):
        view = View()
        view.values[0, 1] = [-1., -1., -1.]
        report = read_existing_articulation_materials(view, context_tag='CPU_FAKE_ONLY')
        self.assertEqual(report['raw_material_slots_including_unresolved_padding'][0][1], [-1., -1., -1.])
        self.assertIsNone(report['padding_mask'])

    def test_backend_failure_not_recast_as_zero(self):
        view = View()
        with patch.object(view, 'get_material_properties', side_effect=RuntimeError('no backend')):
            with self.assertRaisesRegex(RuntimeError, 'no backend'):
                read_existing_articulation_materials(view, context_tag='CPU_FAKE_ONLY')

    def test_numpy_frontend_supported(self):
        class Wrapper:
            def numpy(self):
                return View().values
        report = read_existing_articulation_materials(View(Wrapper()), context_tag='CPU_FAKE_ONLY')
        self.assertEqual(report['count'], 1)

    def test_no_context_rejected_before_getter(self):
        view = View()
        with self.assertRaises(ValueError):
            read_existing_articulation_materials(view, context_tag='')
        self.assertEqual(view.calls, 0)


if __name__ == '__main__':
    unittest.main()