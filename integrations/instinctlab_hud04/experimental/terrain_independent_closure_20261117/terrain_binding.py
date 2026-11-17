"""Hash-bound actual matched-terrain crop. CPU algebra only, NOT runtime cooking.

Raw STL frame and function-shifted frame are EXPLICIT diagnostic layouts.
No source motion/root/quaternion/unit hypothesis is selected here.
"""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import os
import sys
import numpy as np
import trimesh
import yaml

SOLID=Path('/workspace/hud04-solid-geometry-20261117')
sys.path.insert(0,str(SOLID))
from terrain_shape_query import TerrainShapeIndex

ARCHIVE=Path('/workspace/hud04-recovery-fd12fe1')
DATA=Path('/workspace/hud04-original-drive-20261117')
UPSTREAM=ARCHIVE/'upstream/source/instinctlab/instinctlab'
HISTORICAL=Path('/workspace/HUD04_PROCESSED_TERRAIN_INDEPENDENT_CHECK_20261117.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function_from_ast(path,name,scope):
    definitions=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(definitions)!=1:raise ValueError('Ambiguous upstream function')
    node=definitions[0];node.decorator_list=[];node.returns=None
    for arg in node.args.args:arg.annotation=None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),scope)
    return scope[name]


class ActualTerrain:
    def __init__(self):
        if trimesh.__version__!='5.2.1':raise ValueError('Re-reviewed crop dependency required')
        history=json.loads(HISTORICAL.read_text())
        utils=UPSTREAM/'terrains/trimesh/utils.py';generation=UPSTREAM/'terrains/trimesh/mesh_terrains.py'
        for p in [utils,generation]:
            if sha(p)!=history['upstream_source_hashes'][p.name]:raise ValueError('Pinned upstream hash mismatch')
        metadata=DATA/'metadata_kneelClimbStep1.yaml';stl=DATA/'50cmboxes_19_36_49.stl'
        # Original files must agree with the already verified source-only backup.
        manifest=json.loads(Path('/workspace/hud04-motion-checkpoint-20261117/ARCHIVE_MANIFEST.json').read_text())
        for p in [metadata,stl]:
            records=[r for r in manifest['files'] if Path(r['path']).name==p.name]
            if len(records)!=1 or records[0]['sha256']!=sha(p):raise ValueError('Original backup hash mismatch')
        mapping=yaml.safe_load(metadata.read_text())
        if mapping['terrains']!=[dict(terrain_file=stl.name,terrain_id=0)]:raise ValueError('Matched terrain changed')
        task=UPSTREAM/'tasks/shadowing/perceptive/perceptive_env_cfg.py'
        config_tree=ast.parse(task.read_text());generators=[n for n in ast.walk(config_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='FiledTerrainGeneratorCfg']
        if len(generators)!=1 or ast.literal_eval(next(k.value for k in generators[0].keywords if k.arg=='size'))!=(9,12):raise ValueError('Actual task tile size changed')
        cfg_path=UPSTREAM/'terrains/trimesh/mesh_terrains_cfg.py'
        terrain_class=next(n for n in ast.parse(cfg_path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='MotionMatchedTerrainCfg')
        if [ast.unparse(n) for n in terrain_class.bases]!=['SubTerrainBaseCfg']:raise ValueError('Wall inheritance requires review')
        self.config_proof=dict(task_sha256=sha(task),terrain_cfg_sha256=sha(cfg_path),task_size=[9,12],
            terrain_class_direct_base='SubTerrainBaseCfg',wall_mixin_not_inherited=True,
            decorator_returns_early_without_wall_prob=True,runtime_cfg_overrides_observed=False)
        self.inputs={str(p):sha(p) for p in [utils,generation,metadata,stl]}
        scope=dict(np=np,trimesh=trimesh,yaml=yaml,os=os)
        function_from_ast(utils,'crop_terrain_mesh_aabb',scope)
        generate=function_from_ast(generation,'motion_matched_terrain',scope)
        cfg=SimpleNamespace(metadata_yaml=str(metadata),path=str(DATA),size=(9.,12.))
        actual,origin=generate(0.,cfg)
        # Independently retained pre-shift helper, checked against actual AST path.
        sys.path.insert(0,str(ARCHIVE/'integrations/instinctlab_hud04'))
        from terrain_surface_geometry import processed_terrain
        raw,shift,stats=processed_terrain(stl,cfg.size)
        if not np.array_equal(actual.faces,raw.faces) or not np.allclose(actual.vertices,raw.vertices+shift,rtol=0,atol=1e-12) or not np.array_equal(origin,shift):
            raise ValueError('Actual upstream crop/shared-shift disagreement')
        self.mesh=raw;self.triangles=np.asarray(raw.triangles);self.shift=shift;self.stats=stats
        self.lateral_mask=(np.linalg.norm(raw.face_normals,axis=1)>.5)&(abs(raw.face_normals[:,2])<=.1)
        self.lateral_ids=np.flatnonzero(self.lateral_mask)
        if not len(self.lateral_ids):raise ValueError('Expected actual lateral mesh faces')
        self.indices={}
        for layout in ['raw','function_shifted']:
            transform=self.transform(layout)
            for subset in ['all','lateral']:
                triangles=self.triangles if subset=='all' else self.triangles[self.lateral_ids]
                self.indices[layout,subset]=TerrainShapeIndex(triangles,terrain_to_world=transform)

    def transform(self,layout):
        if layout not in ['raw','function_shifted']:raise ValueError('Explicit known algebraic layout required')
        t=np.eye(4)
        if layout=='function_shifted':t[:3,3]=self.shift
        return t

    def original_triangle_id(self,subset,index):
        if subset not in ['all','lateral']:raise ValueError('Explicit subset required')
        return int(index if subset=='all' else self.lateral_ids[index])

    def evidence(self):
        return dict(inputs_sha256=self.inputs,crop_stats=self.stats,triangle_count=len(self.triangles),
            lateral_triangle_count=len(self.lateral_ids),lateral_definition='normal norm > .5 AND abs(normal.z) <= .1; diagnostic ONLY',
            raw_to_terrain_function_shift=self.shift.tolist(),upstream_ast_crop_and_shift_verified=True,
            random_decorator_walls_included=False,static_config_proof=self.config_proof,
            lateral_faces_are_from_matched_stl_not_random_decorator=True,
            generator_grid_placement_included=False,source_root_or_units_inferred=False,
            actual_terrain_runtime_cooking_verified=False,accepted_reference=False,training_ready=False)