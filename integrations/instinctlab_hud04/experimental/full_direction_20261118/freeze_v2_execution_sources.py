"""Archive exact code matching the completed V2 (stopped LP) report.

Source backup only. Does NOT execute or replay any solver/publication/launcher.
"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).parent
report=json.loads((ROOT/'SPAWN_PATH_FIXED_V2'/'FULL_DIRECTION_REPORT.json').read_text())
out=ROOT/'SPAWN_PATH_FIXED_V2'/'CODE_AT_EXECUTION'
out.mkdir(exist_ok=True)
for name,digest in report['code_sha256'].items():
    raw=(ROOT/name).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Current source no longer matches V2 report')
    target=out/name
    if target.exists():
        if target.read_bytes()!=raw:raise ValueError('Backup conflict')
    else:
        with target.open('xb') as f:f.write(raw)
    if target.read_bytes()!=raw:raise ValueError('Code backup readback failed')
print(json.dumps({'backed_up_files':len(report['code_sha256']),'all_match_v2_execution_hashes':True,'solver_replayed':False}))