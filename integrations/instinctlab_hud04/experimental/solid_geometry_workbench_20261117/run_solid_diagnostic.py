"""NEW one-shot CPU diagnostic; no GPU client or historical launch scripts."""
import json
from pathlib import Path
import sys
sys.path.insert(0,'/workspace/hud04-runtime-preflight-20261117')
from bounded_runtime import Budget,supervise,save_diagnostic


if __name__=='__main__':
    output=Path(__file__).resolve().parent/'SOLID_PAIR_DIAGNOSTIC_20261117.json'
    if output.exists():raise FileExistsError('Do not overwrite completed diagnostic')
    result=supervise('solid_worker','run',[{'cpu_only':True}],
        Budget(window_seconds=45,total_seconds=50,iterations=60,numeric_threads=2,max_windows=1))
    save_diagnostic(output,result)
    print(json.dumps(result,indent=2),flush=True)
    if result['status']!='completed':raise SystemExit(1)