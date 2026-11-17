"""Synthetic CPU supervisor tests only. NOT HUD04/Isaac execution."""
import json,sys,time
from pathlib import Path
out=Path(sys.argv[1]);mode=sys.argv[2]
if mode=='sleep':time.sleep(10)
if mode=='error':raise RuntimeError('synthetic fixture exception')
if mode=='flood':sys.stdout.write('x'*(9*1024*1024));sys.stdout.flush();raise SystemExit(0)
report=dict(stage='hud04_import_observed',synthetic_fixture_only=True,cleanup_completed=True,
    snapshot_phases=['post_import','post_clone','post_physics_init'],accepted_reference=False,training_ready=False,
    actual_physx_pair_filters_verified=False,cooked_geometry_verified=False)
if mode=='promoted':report['training_ready']=True
if mode=='partial':report['snapshot_phases']=['post_import']
if mode=='bad_json':(out/'runtime_report.json').write_text('{broken');raise SystemExit(0)
(out/'runtime_report.json').write_text(json.dumps(report))
for phase in ['post_import','post_clone','post_physics_init']:(out/(phase+'.json')).write_text(json.dumps(dict(synthetic_fixture_only=True,phase=phase)))
if mode=='result_then_hang':time.sleep(10)
if mode=='nonzero':raise SystemExit(2)