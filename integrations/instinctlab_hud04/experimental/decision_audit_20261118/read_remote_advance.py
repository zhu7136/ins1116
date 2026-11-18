"""Read-only reconciliation after stale-base publication was blocked."""
import json,os
from pathlib import Path
import requests

ROOT=Path(__file__).parent
BASE='805e620d195bbdd6794009ba368f73890c8af927'
HEAD='4d1a4be8a392bd12759407208b21744395d5a3d1'
API='https://api.github.com/repos/zhu7136/ins1116'

def main():
    headers={'Authorization':'Bearer '+os.environ['github1117'],'Accept':'application/vnd.github+json'}
    def get(path):
        r=requests.get(API+path,headers=headers,timeout=20,allow_redirects=False)
        if r.status_code!=200:raise RuntimeError('GitHub HTTP '+str(r.status_code))
        return r.json()
    d=get('/compare/'+BASE+'...'+HEAD)
    result=dict(base=BASE,observed_head=HEAD,status=d['status'],ahead_by=d['ahead_by'],behind_by=d['behind_by'],total_commits=d['total_commits'],
        commits=[dict(sha=c['sha'],message=c['commit']['message']) for c in d['commits']],
        files=[dict(path=f['filename'],status=f['status'],additions=f['additions'],deletions=f['deletions']) for f in d.get('files',[])],
        mutation_attempted=False)
    with (ROOT/'REMOTE_ADVANCE_READBACK.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':main()