"""Extract pinned official ContactData fields; no network or runtime calls."""
import hashlib
import json
from pathlib import Path

from bs4 import BeautifulSoup


def main():
    here=Path(__file__).resolve().parent
    root=Path('/workspace/hud04-nvidia1073-docs-20261118')
    output=here/'CONTACT_MATERIAL_LINK_API_EVIDENCE.json'
    if output.exists():raise FileExistsError('Evidence already exists; no blind replay')
    receipt=json.loads((root/'physx.receipt.json').read_text())
    raw=(root/'physx.html').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=receipt['sha256']:
        raise ValueError('Official snapshot hash mismatch')
    soup=BeautifulSoup(raw,'html.parser');excerpts=[]
    fields=['ContactData.material0','ContactData.material1','ContactData.face_index0','ContactData.face_index1',
            'ContactEventHeader.collider0','ContactEventHeader.collider1',
            'ContactEventHeader.contact_data_offset','ContactEventHeader.num_contact_data']
    for field in fields:
        anchor='omni.physx.bindings._physx.'+field
        node=soup.find(id=anchor)
        if node is None:raise ValueError('Missing documented field '+anchor)
        text=node.find_parent('dl').get_text(' ',strip=True)
        if len(text)>3000:raise ValueError('Unexpected field excerpt length')
        if field in ('ContactData.material0','ContactData.material1'):
            if 'uint64' not in text or 'intToSdfPath' not in text:
                raise ValueError('Material native ID/decoder contract not found')
        excerpts.append(dict(field=field,url=receipt['url']+'#'+anchor,text=text))
    result=dict(stage='OFFICIAL_DOCUMENTATION_AND_CPU_ADAPTER_NOT_RUNTIME',
        source_receipt=receipt,excerpts=excerpts,
        direct_contact_identity_path='event header actor/collider side -> point offset/index -> same-side material native ID -> decoded material prim path',
        tensor_slot_order_join_required_for_this_identity_path=False,
        no_tensor_slot_mapping_established=True,
        limitation='Material IDs alone do not verify values, combine modes, solver-side overrides or effective friction; no-contact is not separation/coverage proof',
        actual_runtime_material_point_samples=0,actual_gpu_submissions=0,
        runtime_target_api_verified=False,effective_pair_friction=None,
        accepted_reference=False,training_ready=False,
        code_sha256={name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in (
            'contact_material_links_1073.py','native_contact_observer.py',
            'test_contact_material_links_1073.py','collect_contact_material_evidence.py')})
    with output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()