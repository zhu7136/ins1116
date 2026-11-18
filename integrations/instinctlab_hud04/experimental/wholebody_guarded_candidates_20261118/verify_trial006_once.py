"""Reuse existing independent verifier, never solver/old launcher.
Only new trial006 directory receives compatibility aliases/output records.
"""
import hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).parent;TRIAL=ROOT/'WHOLEBODY_TRIAL_006'
OLD=Path('/workspace/hud04-world-candidate-20261118/verify_world_candidate.py')
SHA='ac20ec8ff3406c00d3762f4ffeeee5014ca28023850d6bf2f6f33df8ec8563fd'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if (TRIAL/'INDEPENDENT_WORLD_CANDIDATE_REVIEW.json').exists():raise FileExistsError('Do not replay verification')
    if sha(TRIAL/'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz')!=SHA:raise ValueError('Candidate changed')
    aliases={}
    for a,b in [('RESULT.json','WORLD_CANDIDATE_RESULT.json'),('CORRESPONDENCE_NOT_ACCEPTED.json','CORRESPONDENCE_PROPOSAL_NOT_ACCEPTED.json')]:
        data=(TRIAL/a).read_bytes()
        with (TRIAL/b).open('xb') as f:f.write(data)
        if (TRIAL/b).read_bytes()!=data:raise ValueError('Alias readback')
        aliases[b]=dict(source=a,sha256=sha(TRIAL/a),reason='Identical-byte filename adapter for unchanged prior independent verification function')
    with (TRIAL/'INDEPENDENT_REVIEW_ADAPTER.json').open('x') as f:json.dump(dict(verifier_path=str(OLD),verifier_sha256=sha(OLD),aliases=aliases,artifact_sha256=SHA),f,indent=2)
    spec=importlib.util.spec_from_file_location('existing_independent_world_check',OLD);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.TRIAL=TRIAL;module.OUTPUT=TRIAL/'STEP1_WORLD_CANDIDATE_UNACCEPTED.npz';module.SHA=SHA
    module.main()
if __name__=='__main__':main()