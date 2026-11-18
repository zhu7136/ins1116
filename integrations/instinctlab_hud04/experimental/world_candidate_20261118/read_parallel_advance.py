"""Read-only remote compare; no mutation or task execution."""
import json,os,base64
from pathlib import Path
import requests
ROOT=Path(__file__).parent
API='https://api.github.com/repos/zhu7136/ins1116'
def main():
    headers={'Authorization':'Bearer '+os.environ['github1117'],'Accept':'application/vnd.github+json'}
    def get(path):
        r=requests.get(API+path,headers=headers,timeout=20,allow_redirects=False)
        if r.status_code!=200:raise RuntimeError('GitHub read HTTP'+str(r.status_code))
        return r.json()
    base='70393c676c85888db0bdd92f8ab7e64fe83fa88d';head=get('/git/ref/heads/aria/hud04-asset-camera-probe')['object']['sha']
    compare=get('/compare/'+base+'...'+head)
    report=dict(original_head=base,observed_head=head,ahead_by=compare['ahead_by'],behind_by=compare['behind_by'],
        commits=[dict(sha=c['sha'],message=c['commit']['message']) for c in compare.get('commits',[])],
        files=[{k:f[k] for k in ['filename','status','additions','deletions','sha'] if k in f} for f in compare.get('files',[])])
    texts=[]
    for row in report['files']:
        name=row['filename']
        if name.endswith(('.py','.md','.json')) and ('runtime' in name.lower() or name.endswith('README_zh.md')):
            body=get('/contents/'+name+'?ref='+head)
            if body.get('encoding')=='base64':
                raw=base64.b64decode(body['content'])
                if len(raw)<80000:texts.append('FILE '+name+'\n'+raw.decode('utf-8')+'\n')
    with (ROOT/'PARALLEL_REMOTE_ADVANCE_READBACK.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    with (ROOT/'PARALLEL_REMOTE_CHANGED_TEXT.txt').open('x') as f:f.write('\n'.join(texts))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()