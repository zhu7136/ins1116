"""Corrected per-triangle assertion; retain earlier stopped report/journal."""
from pathlib import Path
import json
import sys
sys.path.insert(0,'/workspace/hud04-runtime-preflight-20261117')
from bounded_runtime import Budget,supervise,save_diagnostic

if __name__=='__main__':
    output=Path(__file__).resolve().parent/'CYLINDER_REFINED_V2_SUPERVISION_20261117.json'
    if output.exists():raise FileExistsError('No overwrite')
    r=supervise('cylinder_verification_worker','run',[dict(cpu_only=True,refined=True,corrected_local_assertion=True)],Budget(window_seconds=45,total_seconds=50,iterations=60,numeric_threads=2,max_windows=1))
    save_diagnostic(output,r);print(json.dumps(r,indent=2),flush=True)
    if r['status']!='completed':raise SystemExit(1)