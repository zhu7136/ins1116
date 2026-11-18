"""One-shot public-source evidence collection, no Flux/simulator/task APIs.

Creates a new report exclusively. Does not overwrite prior evidence or retry.
Requires GitHub access; no credential needed for these public fixed blobs.
"""
import base64
import hashlib
import json
from pathlib import Path
import re
import requests

BLOB='e8d9b2f8a0a1aed3db14fa65d79db1f153598a1f'
SOURCE_PATH='source/extensions/isaacsim.core.api/python/impl/sensors/rigid_contact_view.py'
HERE=Path(__file__).resolve().parent

def main():
    dest=HERE/'API_SOURCE_EVIDENCE.json'
    if dest.exists():raise FileExistsError('Existing source evidence; inspect instead of replay')
    root='https://api.github.com/repos/isaac-sim/IsaacSim'
    s=requests.Session()
    r=s.get(root+'/git/blobs/'+BLOB,timeout=25);r.raise_for_status();body=r.json()
    assert body['sha']==BLOB and body['encoding']=='base64'
    content=base64.b64decode(body['content'])
    git_hash=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    # API object identity and independently computed Git blob hash are recorded
    # separately. A mismatch must not silently become verified source provenance.
    if git_hash!=BLOB:raise ValueError('Git blob bytes do not match declared identity')
    r=s.get(root+'/contents/'+SOURCE_PATH,params={'ref':'v5.1.0'},timeout=25);r.raise_for_status()
    assert r.json()['sha']==BLOB
    lines=content.decode().splitlines()
    excerpts=[dict(first_line=first,last_line=last,text='\n'.join(lines[first-1:last])) for first,last in [(350,376),(441,466)]]
    historical=HERE.parent/'physx_detached_wrist_v2_20261118/failed_task137/contact_error.json'
    raw=historical.read_bytes();previous=json.loads(raw)
    versions=re.findall(r'omni\.physics\.tensors-([^/]+)/omni/physics/tensors/impl/api\.py','\n'.join(previous['frames']))
    if not versions:raise ValueError('Expected measured runtime package signature missing')
    result=dict(schema='hud04.pinned_contact_api_source_evidence.v1',
        source=dict(repo='isaac-sim/IsaacSim',tag='v5.1.0',path=SOURCE_PATH,git_blob=BLOB,
            git_blob_bytes_independently_verified=True,tag_path_blob_verified=True,
            sha256=hashlib.sha256(content).hexdigest(),source_bytes=len(content),excerpts=excerpts,
            url='https://github.com/isaac-sim/IsaacSim/blob/v5.1.0/'+SOURCE_PATH),
        observed_historical_runtime=dict(evidence=str(historical.relative_to(HERE.parents[3])),
            sha256=hashlib.sha256(raw).hexdigest(),historical_directory_label='failed_task137',
            package_identifiers=sorted(set(versions)),historical_task_failed=True,
            full_target_mirror_image_or_build_verified=False),
        source_supported_contract=dict(method='RigidContactView.get_contact_force_data -> _physics_view.get_contact_data(dt)',
            tuple_order=['normal_forces','points','normals','contact_separations','pair_contacts_count','pair_contacts_start_indices'],
            source_only_not_runtime_compatibility=True,units_sign_offsets_not_certified_by_this_report=True),
        unavailable_evidence=['target-build native event subscription/decoder API','cooked vertices bound to live shape IDs','native contact separation units/sign/offset behavior','same-original-articulation causal experiment'],
        native_documentation_access=dict(domain='docs.omniverse.nvidia.com',status='network permission required; connection reset on attempted read',
            url='https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html'),
        gpu_tasks_submitted=0,simulation_steps=0,motion_optimizer_calls=0,automatic_pair_exclusions=0,
        accepted_reference=False,training_ready=False)
    with dest.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(dest),blob=BLOB,sha256=result['source']['sha256'],historical_packages=versions)))

if __name__=='__main__':main()