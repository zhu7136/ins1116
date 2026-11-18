"""CPU preparation and injected existing-stage reads only. No simulator launcher.

CLI writes a NEW preparation manifest. Runtime functions require host-provided
objects; they never import Isaac/pxr, create a stage, cook shapes or step physics.
"""
import argparse
import hashlib
import json
import math
import numbers
from itertools import islice
from pathlib import Path
from audit import HERE,REPO,read,digest
from native_contact_observer import exact_path,registry_from_audit

BASELINE='f9924ea919eeeabeb8ddca4e51d8463d0f2fb18c'

def preparation():
    audit=read(HERE/'AUDIT.json')
    for path,expected in audit['input_files_sha256'].items():
        if digest(REPO/path)!=expected:raise ValueError('Frozen input changed: '+path)
    image_path=HERE.parent/'physx_detached_wrist_v2_20261118/measured_task138/TASK133_DUPLICATE_REVIEW.json'
    historical=read(image_path)['task']
    return dict(schema='hud04.runtime_observation_preparation.v1',baseline_commit=BASELINE,
        preparation_only=True,source_files_sha256={n:digest(HERE/n) for n in
            ['AUDIT.json','contact_report_1073.py','native_contact_observer.py','NVIDIA1073_API_EVIDENCE.json']},
        verified_historical_input_files_sha256=audit['input_files_sha256'],
        historical_image=dict(image_id=historical['imageId'],image_version=historical['imageVersion'],
            evidence_path=str(image_path.relative_to(REPO)),evidence_sha256=digest(image_path),
            immutable_image_digest=None,current_runtime_identity_verified=False),
        original_generated_usd_bytes_sha256=None,original_generated_usd_status='NOT_RETAINED; JSON snapshots are NOT USD file hashes',
        exact_expected_pairs=registry_from_audit(audit),
        current_runtime_config_snapshot=None,existing_authorized_runtime_verified=False,
        execution_status='WAITING_RUNTIME_IDENTITY_AND_AUTHORIZATION',can_start_new_process=False,
        can_submit_gpu=False,planned_limits=dict(host_steps=8,wall_seconds=30,events=256,points=4096),
        frozen_candidate_sha256=audit['frozen_candidate_sha256'],
        gpu_tasks_submitted=0,physics_steps_executed=0,ppo_updates=0,
        accepted_reference=False,training_ready=False)

def scalar_or_small_array(value,limit=64,_depth=0,_budget=None):
    if _budget is None:_budget=[256]
    _budget[0]-=1
    if _budget[0]<0 or _depth>4:raise ValueError('Metadata nesting/node budget exceeded')
    if value is None or isinstance(value,(str,bool)):return value
    if isinstance(value,numbers.Integral):return int(value)
    if isinstance(value,numbers.Real):
        if not math.isfinite(value):raise ValueError('Nonfinite metadata')
        return float(value)
    items=list(islice(iter(value),limit+1))
    if len(items)>limit:raise ValueError('Metadata array exceeds bound')
    return [scalar_or_small_array(v,limit,_depth+1,_budget) for v in items]

def attempt(fn):
    try:return dict(status='OBSERVED',value=fn())
    except Exception as exc:return dict(status='UNKNOWN_OR_ERROR',value=None,error=type(exc).__name__+': '+str(exc)[:300])

def prim_metadata(prim):
    if not prim or not prim.IsValid():raise ValueError('Exact prim unavailable')
    attrs=list(islice(iter(prim.GetAttributes()),129))
    rels=list(islice(iter(prim.GetRelationships()),65))
    if len(attrs)>128 or len(rels)>64:raise ValueError('Prim property budget exceeded')
    out={}
    for a in attrs:
        name=str(a.GetName())
        # Mesh arrays are NOT live cooked geometry; do not copy unbounded arrays.
        if name in ('points','faceVertexCounts','faceVertexIndices','normals','velocities') or name.startswith('primvars:'):
            out[name]=dict(status='NOT_READ_LARGE_OR_NONCOOKED_ARRAY',value=None);continue
        def record(a=a):
            return dict(value=scalar_or_small_array(a.Get()),authored=bool(a.HasAuthoredValueOpinion()),
                time_samples=int(a.GetNumTimeSamples()),time_code='USD_DEFAULT_NOT_CURRENT_PHYSICS_STATE')
        out[name]=attempt(record)
    relationships={}
    for rel in rels:
        def targets(rel=rel):
            values=list(islice(iter(rel.GetTargets()),65))
            if len(values)>64:raise ValueError('Relationship target budget exceeded')
            return [str(v) for v in values]
        relationships[str(rel.GetName())]=attempt(targets)
    return dict(type=str(prim.GetTypeName()),instance_proxy=bool(prim.IsInstanceProxy()),
        attributes=out,relationships=relationships,composed_USD_metadata_only=True,
        final_solver_filter_verified=False,cooked_geometry_verified=False)

def existing_stage_snapshot(stage,settings,bindings,*,paths):
    """Read caller's exact existing stage; no discovery, Apply/Set/Load or export.

    Settings are read through constants supplied by target native bindings. An
    unavailable setting remains unknown, never False. Explicit paths must include
    original actors/colliders/joints, articulation root and scene/group prims.
    """
    if not isinstance(paths,(list,tuple)) or not 1<=len(paths)<=32 or len(set(paths))!=len(paths):
        raise ValueError('One to32 unique exact paths required')
    for path in paths:exact_path(path)
    observed_settings={}
    for name in ('SETTING_COLLISION_APPROXIMATE_CYLINDERS','SETTING_COLLISION_APPROXIMATE_CONES'):
        def setting(name=name):
            key=getattr(bindings,name)
            value=settings.get(key)
            if type(value) is not bool:raise ValueError('Boolean setting unavailable; not coerced')
            return dict(key=key,value=value)
        observed_settings[name]=attempt(setting)
    return dict(schema='hud04.existing_stage_metadata.v1',
        settings=observed_settings,prims={p:attempt(lambda p=p:prim_metadata(stage.GetPrimAtPath(p))) for p in paths},
        snapshot_source='CALLER_EXISTING_STAGE; no runtime created by collector',
        stage_id_independently_verified=False,cpu_gpu_pipeline_independently_verified=False,
        raw_usd_file_hash=None,live_cooked_shape_handle=None,
        global_filter_group_discovery_complete=False,threshold_effectiveness_verified=False,
        final_solver_filter_verified=False,cooked_equivalence_verified=False,
        accepted_reference=False,training_ready=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',type=Path,required=True)
    args=parser.parse_args();manifest=preparation()
    with args.prepare.open('x') as f:json.dump(manifest,f,indent=2,ensure_ascii=False,allow_nan=False)
    print('Preparation saved; NO runtime started; '+manifest['execution_status'])