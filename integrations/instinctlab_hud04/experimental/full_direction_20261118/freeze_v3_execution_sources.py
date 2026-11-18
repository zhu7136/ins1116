"""Immutable code backup matching the completed bounded IPM stop; no solver."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).parent
report=json.loads((ROOT/'GLOBAL_IPM_V3'/'FULL_DIRECTION_REPORT.json').read_text())
out=ROOT/'GLOBAL_IPM_V3'/'CODE_AT_EXECUTION';out.mkdir(exist_ok=True)
for name,digest in report['code_sha256'].items():
    raw=(ROOT/name).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Source differs from V3 execution')
    target=out/name
    if target.exists():
        if target.read_bytes()!=raw:raise ValueError('Backup conflict')
    else:
        with target.open('xb') as f:f.write(raw)
    if target.read_bytes()!=raw:raise ValueError('Source backup readback failed')
print(json.dumps({'files':len(report['code_sha256']),'all_match_v3_execution_hashes':True,'solver_replayed':False}))