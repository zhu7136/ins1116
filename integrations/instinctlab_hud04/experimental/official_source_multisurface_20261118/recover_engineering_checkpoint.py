"""Read current add-only parallel checkpoint; never overwrite/publish/launch."""
import base64,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,'/workspace/hud04-physx-pair-20261118');import publish_and_submit_once as s
OUT=Path(__file__).parent/'PARALLEL_ENGINEERING';BASE='079425dba70333867d6ec339a4a1750294cd6bd8';HEAD='2c7f6a3d3f31a24b993bd7d1a47078c01c9a4cb7'
def main():
    OUT.mkdir(exist_ok=False)
    compare=s.gh('GET','/compare/'+BASE+'...'+HEAD)
    if compare['status']!='ahead' or any(r['status']!='added' for r in compare['files']):raise ValueError('Review divergence')
    tree=s.gh('GET','/git/trees/'+HEAD+'?recursive=1')
    assert not tree.get('truncated');allblobs={r['path']:r for r in tree['tree'] if r['type']=='blob'};records=[]
    prefix='integrations/instinctlab_hud04/experimental/engineering_reference_v1/'
    for entry in compare['files']:
        p=entry['filename'];r=allblobs[p]
        if not p.startswith(prefix) or r['size']>3000000:raise ValueError('Unexpected publication')
        b=base64.b64decode(s.gh('GET','/git/blobs/'+r['sha'])['content']);assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==r['sha']
        target=OUT/p[len(prefix):];target.parent.mkdir(exist_ok=True,parents=True)
        with target.open('xb') as f:f.write(b)
        records.append(dict(path=p,git_sha1=r['sha'],sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
    result=dict(base=BASE,head=HEAD,recovered=records,new_gpu_tasks=0,remote_modified=False)
    out=OUT/'RECOVERY.json';out.write_text(json.dumps(result,indent=2));assert json.loads(out.read_text())==result
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()