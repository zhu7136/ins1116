"""Extract bounded official API excerpts from already downloaded HTML snapshots.

Usage: python collect_nvidia1073_evidence.py SNAPSHOT_DIRECTORY OUTPUT_JSON
No credentials, network, simulation or native interface calls.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from bs4 import BeautifulSoup

SECTIONS={
    'physx':[
        'omni.physx.bindings._physx.ContactEventHeader',
        'omni.physx.bindings._physx.ContactData.separation',
        'omni.physx.bindings._physx.IPhysxSimulation.subscribe_contact_report_events',
        'omni.physx.bindings._physx.IPhysxSimulation.subscribe_full_contact_report_events',
        'omni.physx.bindings._physx.PhysXCooking.request_convex_collision_representation',
        'omni.physx.bindings._physx.PhysxConvexMeshData',
        'omni.physx.bindings._physx.PhysxConvexMeshPolygon'],
    'tensors':['omni.physics.tensors.impl.api.RigidContactView.get_contact_data']}

def collect(root):
    pages=[]
    for name,ids in SECTIONS.items():
        receipt=json.loads((root/(name+'.receipt.json')).read_text())
        data=(root/(name+'.html')).read_bytes()
        assert hashlib.sha256(data).hexdigest()==receipt['sha256']
        assert receipt['status']==200
        assert receipt['url'].startswith('https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/')
        soup=BeautifulSoup(data,'html.parser');excerpts=[]
        for anchor in ids:
            node=soup.find(id=anchor)
            if node is None:raise ValueError('Missing official API anchor: '+anchor)
            text=node.find_parent('dl').get_text('\n',strip=True)
            excerpts.append(dict(anchor=anchor,url=receipt['url']+'#'+anchor,
                text=text[:4200],excerpt_truncated=len(text)>4200))
        pages.append(dict(**receipt,excerpts=excerpts))
    return dict(schema='hud04.nvidia_1073_documentation_evidence.v1',
        collected_at_utc=datetime.now(timezone.utc).isoformat(),
        snapshot_hash_not_immutable_upstream_revision=True,pages=pages,
        confirmed_source_contracts=[
            'IPhysxSimulation.subscribe_contact_report_events: two buffers; returns subscription holder; buffers live one simulation step',
            'subscribe_full_contact_report_events: distinct three-buffer API including friction anchors',
            'Header actor/collider IDs uint64; documented decoder PhysicsSchemaTools::intToSdfPath',
            'Header stage_id and proto_index0/1 exist; 0xFFFFFFFF means not instanced',
            'ContactData.separation is a contact separation value; sign convention/units/offset adjustment not certified by this entry',
            'Tensor get_contact_data(dt) documents six arrays; dt converts impulses to forces',
            'PhysXCooking.request_convex_collision_representation(stage_id,collision_prim_id,run_asynchronously,on_result) exists; returns task and convex mesh data'],
        documentation_caveats=[
            'Tensor returns summary says five but enumerates six, corroborated by previously pinned IsaacSim source; do not accept five buffers',
            'Header decoder notation is C++ style; Python attribute intToSdfPath must be checked on supplied schema tools, not assumed runtime verified',
            'Cooking request is not a guaranteed passive read of already-live solver shape: cache/task behavior, transforms/scale, units and subshape mapping require runtime evidence',
            'No report threshold or ContactReportAPI enablement was changed or independently read back',
            '107.3 family docs do not independently prove package107.3.26 target image support'],
        runtime_evidence_still_required=[
            'target-build native capabilities, stage identity, actual subscription and holder lifetime',
            'existing contact-report enablement/threshold/buffer capacity and bounded time coverage',
            'same original articulation exact collider pairs and final filtering state',
            'live cooked geometry coordinate/scale/subshape equivalence and signed separation semantics',
            'authorized causal experiment without changing unrelated physics context'],
        gpu_tasks_submitted=0,simulation_steps=0,ppo_updates=0,motion_optimizer_calls=0,
        automatic_pair_exclusions=0,accepted_reference=False,training_ready=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('snapshot_directory',type=Path);p.add_argument('output_json',type=Path);a=p.parse_args()
    evidence=collect(a.snapshot_directory)
    with a.output_json.open('x') as f:json.dump(evidence,f,ensure_ascii=False,indent=2)
    print('Saved official107.3 evidence for',sum(len(x['excerpts']) for x in evidence['pages']),'API sections')