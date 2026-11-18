"""NEW read-only reconciliation after first publisher stopped before intent."""
import json,os,base64
from pathlib import Path
import requests
ROOT=Path(__file__).parent;API='https://api.github.com/repos/zhu7136/ins1116'
def main():
    headers={'Authorization':'Bearer '+os.environ['github1117'],'Accept':'application/vnd.github+json'}
    def get(p):
        r=requests.get(API+p,headers=headers,timeout=20,allow_redirects=False)
        if r.status_code!=200:raise RuntimeError('GitHub '+str(r.status_code))
        return r.json()
    old='ee8416303d1dd471b05cc6ae6702e5f52008d554';head=get('/git/ref/heads/aria/hud04-asset-camera-probe')['object']['sha'];c=get('/compare/'+old+'...'+head)
    report=dict(previous_observed_head=old,observed_head=head,ahead_by=c['ahead_by'],behind_by=c['behind_by'],
        commits=[dict(sha=x['sha'],message=x['commit']['message']) for x in c.get('commits',[])],
        files=[{k:x[k] for k in ['filename','status','additions','deletions','sha'] if k in x} for x in c.get('files',[])])
    texts=[]
    for row in report['files']:
        if row['filename'].endswith('.md'):
            body=get('/contents/'+row['filename']+'?ref='+head)
            if body.get('encoding')=='base64':texts.append('FILE '+row['filename']+'\n'+base64.b64decode(body['content']).decode())
    with (ROOT/'PARALLEL_REMOTE_ADVANCE_V2.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    with (ROOT/'PARALLEL_REMOTE_CHANGED_TEXT_V2.txt').open('x') as f:f.write('\n'.join(texts))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()