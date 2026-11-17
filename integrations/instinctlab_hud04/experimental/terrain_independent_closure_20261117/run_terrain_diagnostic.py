from pathlib import Path
import json
import sys
sys.path.insert(0,'/workspace/hud04-runtime-preflight-20261117')
from bounded_runtime import Budget,supervise,save_diagnostic

if __name__=='__main__':
    output=Path(__file__).resolve().parent/'ACTUAL_TERRAIN_SUPERVISION_20261117.json'
    if output.exists():raise FileExistsError('No overwrite')
    result=supervise('terrain_worker','run',[dict(cpu_only=True)],Budget(window_seconds=45,total_seconds=50,iterations=100,numeric_threads=2,max_windows=1))
    save_diagnostic(output,result);print(json.dumps(result,indent=2),flush=True)
    if result['status']!='completed':raise SystemExit(1)