"""Bind actual recovered USD joint metadata, without inventing PhysX exclusions."""
import hashlib,json,itertools,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).parent
EVIDENCE=Path('/workspace/hud04-runtime-progress-20261117/FAILCHAN_FAILURE_EVIDENCE')
INVENTORY=Path('/workspace/hud04-runtime-progress-20261117/RECOVERED_RUNTIME_INVENTORY_REVIEW_20261117.json')
MODEL=Path('/workspace/hud04-recovery-bf4a057/flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bind_row(joint,robot_root,parent,child):
    required={robot_root+'/'+parent,robot_root+'/'+child}
    bodies=joint.get('body0',[])+joint.get('body1',[])
    if len(bodies)!=2 or set(bodies)!=required:raise ValueError('Exact composed joint-body binding required')
    c=joint['collision_enabled'];e=joint['joint_enabled'];x=joint['exclude_from_articulation']
    if not c.get('present') or type(c.get('value')) is not bool or not e.get('present') or e.get('value') is not True or not x.get('present') or x.get('value') is not False:raise ValueError('Incomplete active articulation joint evidence')
    return dict(usd_joint=joint['path'],body0=joint['body0'],body1=joint['body1'],resolved_joint_collision_enabled=c['value'],
        collision_enabled_authored=bool(c['authored']),resolved_from_schema_fallback=not c['authored'],
        metadata_intent='joint_body_collision_disabled' if not c['value'] else 'joint_body_collision_enabled',
        actual_physx_pair_filter_verified=False,pair_exclusion_authorized=False)
def collect():
    inventory=json.loads(INVENTORY.read_text())
    if inventory['task_id']!='TASK_20261117_366' or inventory['terminal_status']!='6' or inventory['worker_result']['return_code']!=-11:raise ValueError('Keep original task failure')
    if sha(MODEL)!=inventory['source_model_sha256']:raise ValueError('Original model identity')
    xml=ET.parse(MODEL).getroot();links={l.get('name'):len(l.findall('collision')) for l in xml.findall('link')}
    shapes=[(l,k) for l,n in links.items() for k in range(n)]
    pairs=list(itertools.combinations(shapes,2));body_joints={frozenset([j.find('parent').get('link'),j.find('child').get('link')]):j for j in xml.findall('joint')}
    snapshots={}
    for name,receipt in inventory['snapshots'].items():
        path=EVIDENCE/name
        if sha(path)!=receipt['file_sha256']:raise ValueError('Recovered snapshot identity')
        s=json.loads(path.read_text());root=s['robot_root'];joints={j['path'].removeprefix(root+'/joints/'):j for j in s['joints']}
        rows=[]
        for (la,ia),(lb,ib) in pairs:
            joint=body_joints.get(frozenset([la,lb]))
            if la==lb or joint is None:continue
            name_j=joint.get('name')
            if name_j not in joints:raise ValueError('Missing exact joint')
            row=bind_row(joints[name_j],root,joint.find('parent').get('link'),joint.find('child').get('link'))
            row.update(first=la+'#'+str(ia),second=lb+'#'+str(ib),urdf_joint=name_j)
            rows.append(row)
        if len(rows)!=28:raise ValueError('Original direct pair inventory')
        snapshots[name]=dict(file_sha256=sha(path),runtime_phase=s['runtime_phase'],rows=rows)
    orders=[[{k:v for k,v in r.items() if k!='runtime_phase'} for r in s['rows']] for s in snapshots.values()]
    if any(x!=orders[0] for x in orders[1:]):raise ValueError('Metadata changed across phases; unresolved')
    return dict(stage='REAL_failed_task_USD_joint_metadata_bound_to_nominal_pairs_NOT_runtime_filter_acceptance',
        task_id=inventory['task_id'],actual_commit=inventory['actual_commit'],platform_status='6',worker_return_code=-11,
        model_sha256=sha(MODEL),snapshots=snapshots,total_nominal_shapes=len(shapes),all_shape_pairs_retained=len(pairs),
        same_rigid_link_pairs=sum(a[0]==b[0] for a,b in pairs),direct_joint_shape_pairs=28,
        unique_joint_body_bindings=len({r['urdf_joint'] for r in orders[0]}),
        all_three_phases_same_pair_metadata=True,
        metadata_disabled_shape_pairs=sum(not r['resolved_joint_collision_enabled'] for r in orders[0]),
        actual_physx_pair_filters_verified=False,automatic_pair_exclusions=0,collision_constraints_removed=0,
        pair_exclusion_authorized=False,cooked_geometry_verified=False,source_world_unit_alignment_proven=False,
        observation_completed=False,motion_solver_calls=0,new_motion_files=0,new_gpu_submissions=0,ppo_updates=0,
        accepted_reference=False,training_ready=False)
def main():
    out=ROOT/'BOUND_PAIR_POLICY_EVIDENCE.json'
    if out.exists():raise FileExistsError('No replay')
    r=collect()
    with out.open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
    if json.loads(out.read_text())!=r:raise ValueError('Local readback')
    print(json.dumps({k:v for k,v in r.items() if k!='snapshots'},indent=2))
if __name__=='__main__':main()