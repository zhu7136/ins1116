"""New single-use CPU-only supervisor. Does not import any service client."""
from pathlib import Path
import json
import sys
sys.path.insert(0,'/workspace/hud04-runtime-preflight-20261117')
from bounded_runtime import Budget,supervise,save_diagnostic

if __name__=='__main__':
    out=Path(__file__).resolve().parent/'ACTUAL_TERRAIN_BINDING_DIAGNOSTIC_20261117.json'
    if out.exists():raise FileExistsError('One shot; preserve completed/failed evidence')
    report=supervise('binding_worker','run',[dict(mode='source'),dict(mode='target')],
        Budget(window_seconds=55,total_seconds=115,iterations=100,numeric_threads=2,max_windows=2))
    save_diagnostic(out,report)
    print(json.dumps(report,indent=2),flush=True)
    if report['status']!='completed':raise SystemExit(1)