"""New algorithm trial: HiGHS IPM on identical 997-frame objective/hard rows.

NOT a transient retry or enlarged time/iteration budget. V2 stop stays intact.
"""
import os,json,hashlib
from pathlib import Path
ROOT=Path(__file__).parent
report=json.loads((ROOT/'SPAWN_PATH_FIXED_V2'/'FULL_DIRECTION_REPORT.json').read_text())
if report['actual_lp_calls']!=1 or report['accepted_steps']!=0 or report['history'][0]['lp_iterations']!=10000:raise ValueError('Expected original bounded-simplex stop missing')
if report['lp_method']!='highs' or report['termination']!='LP_stopped_status_1':raise ValueError('Not the reviewed algorithm stop')
for name,digest in report['code_sha256'].items():
    path=ROOT/'SPAWN_PATH_FIXED_V2'/'CODE_AT_EXECUTION'/name
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('V2 source archive no longer matches stop report')
os.environ['HUD04_FULL_DIRECTION_EXECUTION']='global_ipm_v3'
from run_full_direction import main,worker

if __name__=='__main__':main()