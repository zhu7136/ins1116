"""Static source audit and passive 107.3 material getter; never starts physics.

Array slot indices are NOT collider identities. Neither successful collection
nor static declarations establish effective pair friction or contact feasibility.
"""
import ast
import hashlib
from pathlib import Path

import numpy as np


def source_record(path):
    path = Path(path)
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
            'tree': ast.parse(raw, filename=str(path))}


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + '.' + node.attr
    return ''


def named_class(tree, name):
    matches = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError('Expected unique class: ' + name)
    return matches[0]


def audit_source(upstream, plane, terrain, asset):
    sources = {k: source_record(v) for k, v in dict(
        upstream=upstream, plane=plane, terrain=terrain, asset=asset).items()}
    scene = named_class(sources['upstream']['tree'], 'PerceptiveShadowingSceneCfg')
    declarations = [n.value for n in scene.body if isinstance(n, ast.Assign)
                    and any(dotted(t) == 'terrain' for t in n.targets)]
    if len(declarations) != 1 or not isinstance(declarations[0], ast.Call):
        raise ValueError('Terrain declaration changed')
    material = [k.value for k in declarations[0].keywords if k.arg == 'physics_material']
    if len(material) != 1 or not isinstance(material[0], ast.Call):
        raise ValueError('Expected explicit terrain material')
    if dotted(material[0].func) != 'sim_utils.RigidBodyMaterialCfg':
        raise ValueError('Unexpected material constructor')
    values = {k.arg: ast.literal_eval(k.value) for k in material[0].keywords}
    required = {'static_friction', 'dynamic_friction', 'friction_combine_mode'}
    if not required <= values.keys():
        raise ValueError('Explicit friction declarations missing')
    # Verify the inspected HUD04 loop actually disables the randomization term.
    plane_cls = named_class(sources['plane']['tree'], 'HUD04PerceptivePlaneEnvCfg')
    loops = [n for n in ast.walk(plane_cls) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == 'name'
             and isinstance(n.iter, (ast.Tuple, ast.List))]
    disabled = False
    for loop in loops:
        names = ast.literal_eval(loop.iter)
        if 'physics_material' not in names:
            continue
        for statement in loop.body:
            if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                continue
            call = statement.value
            disabled |= (dotted(call.func) == 'setattr' and len(call.args) == 3
                         and dotted(call.args[0]) == 'self.events'
                         and dotted(call.args[1]) == 'name'
                         and isinstance(call.args[2], ast.Constant)
                         and call.args[2].value is None)
    if not disabled:
        raise ValueError('HUD04 material randomization disable not established')
    terrain_cls = named_class(sources['terrain']['tree'], 'HUD04PerceptiveTerrainEnvCfg')
    restore = [n for n in ast.walk(terrain_cls) if isinstance(n, ast.Assign)
               and any(dotted(t) == 'self.scene.terrain' for t in n.targets)]
    if len(restore) != 1 or ast.unparse(restore[0].value) != 'PerceptiveShadowingSceneCfg().terrain.copy()':
        raise ValueError('Terrain copy path changed')
    calls = [n for n in ast.walk(sources['asset']['tree']) if isinstance(n, ast.Call)
             and dotted(n.func) == 'sim_utils.UrdfFileCfg']
    if len(calls) != 1:
        raise ValueError('Expected unique robot importer declaration')
    explicit_material = any(k.arg == 'physics_material' for k in calls[0].keywords)
    return {
        'stage': 'STATIC_SOURCE_DECLARATIONS_ONLY_NOT_RUNTIME',
        'source_files': {k: {f: v[f] for f in ('path', 'sha256')} for k, v in sources.items()},
        'terrain_material_explicit_keywords': values,
        'hud04_startup_material_randomization_disabled': disabled,
        'terrain_restored_from_same_upstream_scene': True,
        'robot_importer_explicit_physics_material_keyword': explicit_material,
        'robot_shape_materials': None,
        'effective_pair_static_friction': None,
        'effective_pair_dynamic_friction': None,
        'runtime_material_binding_verified': False,
        'limitations': ['No SDK import, config construction, stage or tensor read occurred',
                        'No inference of robot defaults from missing importer keyword',
                        'Does not prove absence of other overrides or establish combine priority',
                        'Source declarations do not certify runtime values, normals or modes'],
        'accepted_reference': False, 'training_ready': False,
    }


def read_existing_articulation_materials(view, *, context_tag):
    """Read caller-owned EXISTING view only; no create/step/set/subscription.

    context_tag is caller-supplied, NOT verified runtime identity/permission.
    Padding and shape order remain unresolved; output cannot bind pair friction.
    Getter schema: pinned NVIDIA107.3 ArticulationView.get_material_properties.
    """
    if not isinstance(context_tag, str) or not 1 <= len(context_tag) <= 256:
        raise ValueError('Bounded nonempty context tag required')
    count, max_shapes = view.count, view.max_shapes
    if (isinstance(count, bool) or isinstance(max_shapes, bool)
            or not isinstance(count, (int, np.integer))
            or not isinstance(max_shapes, (int, np.integer))
            or not 1 <= count <= 8 or not 1 <= max_shapes <= 256):
        raise ValueError('Existing view exceeds read budget or has invalid dimensions')
    raw = view.get_material_properties()  # Exactly one getter; exceptions propagate.
    if hasattr(raw, 'detach'):
        raw = raw.detach().cpu().numpy()
    elif hasattr(raw, 'numpy'):
        raw = raw.numpy()
    array = np.asarray(raw)
    if array.shape != (count, max_shapes, 3) or array.dtype.kind not in 'fi':
        raise ValueError('Unexpected107.3 material tensor shape/type; no inferred reshape')
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite material tensor')
    return {
        'stage': 'CALLER_VIEW_MATERIAL_SLOTS_READ_NOT_COLLIDER_BINDING',
        'caller_context_tag_unverified': context_tag,
        'count': int(count), 'max_shapes': int(max_shapes),
        'column_order': ['static_friction', 'dynamic_friction', 'restitution'],
        'raw_material_slots_including_unresolved_padding': array.copy().tolist(),
        'slot_to_collider_mapping': None,
        'padding_mask': None,
        'terrain_materials': None,
        'combine_modes': None,
        'effective_pair_friction': None,
        'runtime_identity_verified': False,
        'runtime_material_binding_verified': False,
        'accepted_reference': False, 'training_ready': False,
    }