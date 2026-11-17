from pathlib import Path
import json
import sys
sys.path.insert(0,'/workspace/hud04-runtime-preflight-20261117')
from bounded_runtime import Budget,supervise,save_diagnostic

if __name__=='__main__':
    out=Path(__file__).resolve().parent/'INDEPENDENT_VERIFICATION_SUPERVISION_20261117.json'
    if out.exists():raise FileExistsError('Single use')
    r=supervise('verification_worker','run',[dict(cpu_only=True)],Budget(window_seconds=40,total_seconds=45,iterations=100,numeric_threads=2,max_windows=1))
    save_diagnostic(out,r);print(json.dumps(r,indent=2),flush=True)
    if r['status']!='completed':raise SystemExit(1)