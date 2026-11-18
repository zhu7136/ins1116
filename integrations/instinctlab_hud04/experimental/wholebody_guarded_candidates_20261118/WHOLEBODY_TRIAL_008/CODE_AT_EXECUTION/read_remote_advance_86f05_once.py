"""Read-only recovery of concurrent upstream changes; never overwrite workbench."""
import base64,hashlib,json,os
from pathlib import Path
import requests
ROOT=Path(__file__).parent;BASE='e50115ec3fdabc3583224ea060e3054d61631939';HEAD='86f05dffa8696465147d7e3ebf99aff568a84c26'
def main():
    directory=ROOT/'REMOTE_ADVANCE_REVIEW_86f05';directory.mkdir(exist_ok=True)
    session=requests.Session();session.headers.update({'Authorization':'Bearer '+os.environ['github1117'],'Accept':'application/vnd.github+json'})
    def get(p):
        r=session.get('https://api.github.com/repos/zhu7136/ins1116'+p,timeout=20,allow_redirects=False)
        if r.status_code!=200:raise RuntimeError('GitHub status '+str(r.status_code))
        return r.json()
    comparison=get('/compare/'+BASE+'...'+HEAD)
    if comparison['behind_by']!=0:raise ValueError('Divergent remote')
    files={}
    for f in comparison['files']:
        if f['status'] not in ['added','modified']:raise ValueError('Unexpected change kind')
        path=f['filename']
        if not path.startswith('integrations/instinctlab_hud04/experimental/'):raise ValueError('Scope')
        name=Path(path).relative_to('integrations/instinctlab_hud04/experimental').as_posix()
        if Path(path).suffix=='.json':
            files[path]=dict(git_blob=f['sha'],status=f['status'],recovered=False,reason='Large audit output not needed for code review; do not claim local recovery')
            continue
        response=get('/git/blobs/'+f['sha']);content=base64.b64decode(response['content'])
        if not 0<len(content)<250000:raise ValueError('Byte budget')
        (directory/name).parent.mkdir(parents=True,exist_ok=True)
        if (directory/name).exists():
            if (directory/name).read_bytes()!=content:raise ValueError('Existing recovery bytes differ; no overwrite')
        else:
            with (directory/name).open('xb') as output:output.write(content)
        files[path]=dict(local_file=name,git_blob=f['sha'],sha256=hashlib.sha256(content).hexdigest(),status=f['status'])
    with (directory/'REMOTE_READBACK.json').open('x') as output:json.dump(dict(base=BASE,head=HEAD,ahead_by=comparison['ahead_by'],files=files,read_only=True),output,indent=2)
    print(json.dumps(dict(head=HEAD,files=len(files),read_only=True),indent=2))
if __name__=='__main__':main()