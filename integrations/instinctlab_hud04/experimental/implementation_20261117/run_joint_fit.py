"""One CPU-only full-length Step1 fit with a parent deadline and W&B record."""
import hashlib,json,multiprocessing as mp,os,time
from pathlib import Path

ROOT=Path(__file__).parent
INTEGRATION=Path(os.environ.get('HUD04_DELIVERY_ROOT','/workspace/hud04-recovery-bf4a057'))/'integrations/instinctlab_hud04'
MOTION=Path(os.environ.get('HUD04_STEP1_MOTION','/workspace/hud04-original-drive-20261117/kneelClimbStep1-x-0.1-ziwen-retargeted.npz'))
TARGET=INTEGRATION.parent.parent/'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def worker(connection):
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import sys
    import numpy as np
    sys.path.insert(0,str(INTEGRATION))
    import retarget_dry_run as r
    from joint_temporal_fit import fit
    os.environ['HUD04_DELIVERY_ROOT']=str(INTEGRATION.parent.parent)
    r.require_hash(MOTION,r.MOTION_SHA256);r.require_hash(TARGET,r.TARGET_MODEL_SHA256)
    names,model=r.A.C.validate_model(TARGET)
    with np.load(MOTION,allow_pickle=False) as z:
        stored=z['joint_names'].tolist();q=z['joint_pos'].astype(float)[:,[stored.index(n) for n in names]]
        if q.shape!=(997,29) or float(z['framerate'].item())!=90 or len(set(stored))!=29 or set(stored)!=set(names): raise ValueError('Step1 contract')
    lower,upper,vmax=[np.array([model[n][k] for n in names]) for k in ('lower','upper','velocity')]
    out,report=fit(q,lower,upper,vmax)
    result=ROOT/'step1.joint_fit.npz'
    with result.open('xb') as f: np.savez_compressed(f,joint_pos=out,joint_names=np.array(names),framerate=np.array(90))
    with np.load(result,allow_pickle=False) as z: np.testing.assert_array_equal(z['joint_pos'],out)
    r.require_hash(MOTION,r.MOTION_SHA256);r.require_hash(TARGET,r.TARGET_MODEL_SHA256)
    report.update(source_sha256=r.MOTION_SHA256,model_sha256=r.TARGET_MODEL_SHA256,candidate_sha256=digest(result),
        candidate_file=result.name,candidate_is_reference_motion=False,source_data_modified=False,
        source_position_bad_frames=int(np.any((q<lower-1e-6)|(q>upper+1e-6),axis=1).sum()),
        source_overspeed_intervals=int(np.any(np.abs(np.diff(q,axis=0))*90>vmax+1e-6,axis=1).sum()),
        output_position_bad_frames=int(np.any((out<lower-1e-6)|(out>upper+1e-6),axis=1).sum()),
        output_overspeed_intervals=int(np.any(np.abs(np.diff(out,axis=0))*90>vmax+1e-6,axis=1).sum()),
        joint_names=names,missing=['root/pose optimization','terrain and self-collision constraints','auditable support/contact constraints','dynamic feasibility'])
    with (ROOT/'STEP1_JOINT_FIT_RESULT.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    connection.send(dict(status='completed',report='STEP1_JOINT_FIT_RESULT.json'));connection.close()

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--offline-after-audience-failure',action='store_true')
    args=parser.parse_args()
    if (ROOT/'JOINT_FIT_INTENT.json').exists(): raise RuntimeError('One-time trial already intended; inspect existing outputs')
    import wandb
    provenance={k:os.environ[v] for k,v in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if os.environ.get(v)}
    # Explicit target, online mode, and finite initialization time. No silent
    # offline record presented as a live experiment if audience is still wrong.
    run=wandb.init(entity='zhul136-university-of-science-and-technology-of-china',project='ins1116',
        name='hud04-step1-joint-temporal-fit-not-reference',job_type='joint-fit-diagnostic',
        mode='offline' if args.offline_after_audience_failure else 'online',
        config={'_wb_agent':provenance,'source_commit':'2e0506721c1cae6a9bd117899db1b3e48192b0ba',
                'frames':997,'framerate':90,'cpu_only':True,'accepted_reference':False,'training_ready':False},
        settings=wandb.Settings(init_timeout=25))
    with (ROOT/'JOINT_FIT_INTENT.json').open('x') as f:json.dump(dict(run_url=run.url,run_id=run.id,wall_limit_seconds=90,gpu_submissions=0),f)
    recv,send=mp.get_context('spawn').Pipe(duplex=False)
    p=mp.get_context('spawn').Process(target=worker,args=(send,));start=time.monotonic();p.start();send.close()
    try:
        while p.is_alive() and time.monotonic()-start<90:p.join(.05)
        if p.is_alive():p.terminate();p.join(.3)
        if p.is_alive():p.kill();p.join(.7)
        if p.is_alive() or p.exitcode!=0 or not recv.poll(.1):raise RuntimeError('CPU worker failure or deadline; do not retry automatically')
        message=recv.recv()
        report=json.loads((ROOT/message['report']).read_text())
        run.log({k:report[k] for k in ('solver_calls','full_retarget_solver_calls','rms_change_rad','max_change_rad',
            'source_position_bad_frames','source_overspeed_intervals','output_position_bad_frames','output_overspeed_intervals')})
        for k,v in report['residuals'].items():run.summary[k]=v
        run.summary['accepted_reference']=False;run.summary['training_ready']=False
        run.finish()
        # Re-read the server, not just the locally finished run.
        if not args.offline_after_audience_failure:
            check=wandb.Api(timeout=20).run('/'.join(run.path))
            if check.summary.get('output_position_bad_frames')!=report['output_position_bad_frames']:raise RuntimeError('W&B readback mismatch')
        receipt=dict(run_url=run.url,run_id=run.id,server_readback_verified=not args.offline_after_audience_failure,
                     logging_mode='offline_pending_sync_invalid_audience' if args.offline_after_audience_failure else 'online',
                     child_exit_code=p.exitcode,child_reaped=not p.is_alive(),
                     gpu_submissions=0,ppo_updates=0,accepted_reference=False,training_ready=False)
        with (ROOT/'JOINT_FIT_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2)
        print(json.dumps(dict(**receipt,result=report['residuals'],elapsed_seconds=report['elapsed_seconds'],
            rms_change_rad=report['rms_change_rad'],max_change_rad=report['max_change_rad']),indent=2),flush=True)
    except BaseException:
        run.finish(exit_code=1);raise
    finally:recv.close()

if __name__=='__main__':main()