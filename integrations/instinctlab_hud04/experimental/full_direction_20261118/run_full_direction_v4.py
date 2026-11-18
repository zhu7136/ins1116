"""Fresh overlapping-global-objective trial, not replay of global LP stops."""
import os,json,hashlib
from pathlib import Path
ROOT=Path(__file__).parent
for folder,method in (('SPAWN_PATH_FIXED_V2','highs'),('GLOBAL_IPM_V3','highs-ipm')):
    report=json.loads((ROOT/folder/'FULL_DIRECTION_REPORT.json').read_text())
    if report['actual_lp_calls']!=1 or report['accepted_steps']!=0 or report['lp_method']!=method:raise ValueError('Prior bounded-stop evidence differs')
    for name,digest in report['code_sha256'].items():
        if hashlib.sha256((ROOT/folder/'CODE_AT_EXECUTION'/name).read_bytes()).hexdigest()!=digest:raise ValueError('Prior numerical-code archive changed')
os.environ['HUD04_FULL_DIRECTION_EXECUTION']='overlap_v4'
from run_full_direction import main,worker

if __name__=='__main__':main()