"""NEW CPU matrix diagnostic only. No solver, Flux or GPU calls."""
import argparse,hashlib,json,os,re
from pathlib import Path
from bounded_runtime import Budget,supervise,save_diagnostic

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--label',default='v1');args=parser.parse_args()
    if not re.fullmatch(r'v[1-9][0-9]?',args.label):raise ValueError('Explicit diagnostic version label required')
    root=Path(__file__).parent;dest=root/('WINDOW_SYSTEM_REVIEW_'+args.label+'.json');intent=root/('WINDOW_DIAGNOSTIC_INTENT_'+args.label+'.json')
    if args.label=='v1' and (root/'WINDOW_SYSTEM_REVIEW.json').exists():raise FileExistsError('Legacy v1 diagnostic already exists')
    if dest.exists() or intent.exists():raise FileExistsError('Existing attempt; do not blindly replay')
    delivery=Path(os.environ.get('HUD04_DELIVERY_ROOT','/workspace/hud04-recovery-bf4a057'))
    model=delivery/'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
    fit=Path(os.environ.get('HUD04_JOINT_FIT','/workspace/hud04-implementation-20261117/step1.joint_fit.npz'))
    budget=Budget(window_seconds=30,total_seconds=40,iterations=5,numeric_threads=2,max_windows=1)
    code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'named_target.py',root/'window_problem.py',root/'window_worker.py',root/'bounded_runtime.py',Path(__file__))}
    with intent.open('x') as f:json.dump(dict(kind='CPU_software_diagnostic_NOT_training_trial',label=args.label,model=str(model),fit=str(fit),implementation_sha256=code_sha256,solver_calls=0,gpu_submissions=0),f,indent=2)
    result=supervise('window_worker','construct',[dict(model=str(model),fit=str(fit))],budget)
    result['implementation_sha256']=code_sha256
    save_diagnostic(dest,result)
    if json.loads(dest.read_text())!=result:raise ValueError('Diagnostic readback')
    print(json.dumps(result,indent=2),flush=True)
    if result['status']!='completed':raise SystemExit(1)

if __name__=='__main__':main()