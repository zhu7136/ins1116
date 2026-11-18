"""One 15-frame CPU probe of explicit hard nearby-separation constraints.

Not a rerun of timed-out 001: changed mathematical constraints, one LP,
one bounded window, at most three candidate endpoint checks, no full export.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import warnings
import numpy as np
from scipy import sparse
from scipy.optimize import linprog, OptimizeWarning
from material_problem import HERE, REPO, SEED_SHA, problem, perturb, hard_rows
from clearance_rows import build

OUT=HERE/'TRIAL_MATERIAL_002_HARD_CLEARANCE_PROBE'


def put(name,data):
    with (OUT/name).open('x') as f:json.dump(data,f,indent=2,allow_nan=False)


def worker():
    import wandb
    intent=json.loads((OUT/'INTENT.json').read_text())
    run=wandb.init(entity='zhul136-university-of-science-and-technology-of-china',project='ins1116',
                    id=intent['wandb_run_id'],config=intent,name='hud04-material-hard-clearance-002-NOT-ACCEPTED',
                    job_type='cpu-material-constraint-probe',dir=str(OUT),
                    settings=wandb.Settings(disable_git=True,save_code=False))
    start=time.monotonic()
    report=dict(accepted_reference=False,training_ready=False,gpu_submissions=0,ppo_updates=0,
                contact_or_flight_inferred=False,unassigned_support_frames=346,motion_exported=False,
                frames_tested=15,full_997_frame_solve=False,hard_clearance_is_not_physical_acceptance=True,
                backtracking=[])
    def phase(name):
        with (OUT/'PHASES.jsonl').open('a') as f:f.write(json.dumps(dict(phase=name,wall_s=time.monotonic()-start))+'\n')
        print(name,flush=True)
    try:
        m,q,roots,want,w,support,body,guard,kernel=problem();first,last=0,14;n=15;nv=35*n
        piece=q[:n];r=roots[:n];weights=w[:n]
        value,j=m.evaluate(piece,r,True);res=((value-want[:n])*weights).reshape(-1)
        a=sparse.block_diag([sparse.csr_matrix(v*x[:,None]) for v,x in zip(j,weights)],format='csr');na=len(res)
        hard,rhs=hard_rows(m,q,first,last)
        clear,crhs,records=build(body,kernel,m,piece,r);phase('hard_rows_built')
        gaps,gj,grecords,counts=body.evaluate(m,piece,r,True)
        same=np.array([t['kind']=='self' and body.shapes[t['first']].link==body.shapes[t['second']].link for t in grecords],bool)
        movable=np.asarray(gj.getnnz(axis=1)>0).ravel() & ~same
        gl,jl=gaps[movable],gj[movable];ng=len(gl)
        matrix=sparse.vstack([
            sparse.hstack([hard,sparse.csr_matrix((len(rhs),na+ng))]),
            sparse.hstack([clear,sparse.csr_matrix((len(crhs),na+ng))]),
            sparse.hstack([a,-sparse.eye(na),sparse.csr_matrix((na,ng))]),
            sparse.hstack([-a,-sparse.eye(na),sparse.csr_matrix((na,ng))]),
            sparse.hstack([-1000*jl,sparse.csr_matrix((ng,na)),-sparse.eye(ng)])],format='csr')
        b=np.r_[rhs,crhs,-res,res,1000*(gl-body.margin)]
        trust=np.tile(np.r_[np.full(3,.015),np.full(3,.04),np.full(29,.08)],n)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=OptimizeWarning)
            lp=linprog(np.r_[np.zeros(nv),np.ones(na+ng)],A_ub=matrix,b_ub=b,
                       bounds=np.c_[-trust,trust].tolist()+[(0,None)]*(na+ng),method='highs',
                       options={'time_limit':2.,'maxiter':12000,'threads':2,'parallel':False})
        report.update(lp_calls=1,lp_status=int(lp.status),lp_message=lp.message,nearby_hard_rows=len(crhs),
                      material_rows=na,geometry_penalty_rows=ng,initial_geometry_projection_counts=counts)
        phase('one_LP_complete')
        if lp.success:
            step=lp.x[:nv].reshape(n,35);old=float(abs(res).sum()+1000*np.maximum(body.margin-gaps,0).sum())
            old_state=guard.scan(m,piece,r,np.arange(n));phase('initial_independent_geometry_complete')
            for alpha in [1.,.25,.0625]:
                if time.monotonic()-start>55:report['backtracking_budget_stop']=True;break
                tq,tr=piece+alpha*step[:,6:],perturb(r,alpha*step[:,:6])
                if np.max(hard@(alpha*step.ravel())-rhs)>1e-6:raise ValueError('Original hard limits violated')
                if np.max(clear@(alpha*step.ravel())-crhs)>1e-6:raise ValueError('New hard linear separation rows violated')
                ngap,_,_=body.evaluate(m,tq,tr,False)
                new=float(abs((m.evaluate(tq,tr)-want[:n])*weights).sum()+1000*np.maximum(body.margin-ngap,0).sum())
                state=guard.scan(m,tq,tr,np.arange(n));passed,check=guard.compare(old_state,state)
                detail=dict(alpha=alpha,old_merit=old,new_merit=new,merit_improved=new<old-1e-9,
                            endpoint_guard=check,engineering_step_accepted=bool(passed and new<old-1e-9))
                report['backtracking'].append(detail);phase('candidate_endpoint_check_complete')
                if detail['engineering_step_accepted']:
                    np.savez_compressed(OUT/'WINDOW0_14_UNACCEPTED.npz',joint_pos=tq,joint_names=np.array(m.names),
                                        root_transforms_w=tr,frame_indices=np.arange(n),framerate=np.array(90),
                                        accepted_reference=np.array(False),training_ready=np.array(False),
                                        artifact_role=np.array('PARTIAL_15_FRAME_NUMERICAL_PROBE_NOT_TRAINING_INPUT'))
                    break
        report.update(accepted_updates=sum(x['engineering_step_accepted'] for x in report['backtracking']),
                      wall_s=time.monotonic()-start,seed_sha256=SEED_SHA,original_geometry_unchanged=True)
        guard.verify_inputs();put('RESULT.json',report)
        run.summary.update({k:v for k,v in report.items() if isinstance(v,(int,float,bool,str))})
        run.finish()
    except BaseException:
        run.finish(exit_code=1);raise


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--worker':return worker()
    tests=Path('/workspace/hud04-material-trajectory-pretests12.log').read_text()
    if 'Ran 12 tests' not in tests or not tests.rstrip().endswith('OK'):raise ValueError('12 software pretests required')
    OUT.mkdir(exist_ok=False)
    config=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                user_authority='Continue finite CPU reference reconstruction; no GPU or PPO',
                source_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
                seed_sha256=SEED_SHA,first_frame=0,last_frame=14,maximum_lp_calls=1,
                maximum_endpoint_checks=3,worker_wall_limit_s=80,automatic_retries=0,
                body_dof=29,head_dof=0,accepted_reference=False,training_ready=False)
    provenance={k:os.environ[e] for k,e in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if e in os.environ}
    if provenance:config['_wb_agent']=provenance
    config['wandb_run_id']=uuid.uuid4().hex[:8];put('INTENT.json',config)
    snap=OUT/'CODE_AT_EXECUTION';snap.mkdir()
    for p in HERE.glob('*.py'):(snap/p.name).write_bytes(p.read_bytes())
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    with (OUT/'EXECUTION.log').open('x') as f:
        child=subprocess.Popen([sys.executable,str(Path(__file__)),'--worker'],env=env,stdout=f,stderr=subprocess.STDOUT)
        timedout=False
        try:child.wait(timeout=80)
        except subprocess.TimeoutExpired:
            timedout=True;child.terminate()
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2)
    receipt=dict(child_exit=child.returncode,child_reaped=child.poll() is not None,timed_out=timedout,
                 result_exists=(OUT/'RESULT.json').exists(),gpu_submissions=0,ppo_updates=0)
    put('SUPERVISION.json',receipt);print(json.dumps(receipt),flush=True)
    if child.returncode:raise SystemExit(1)


if __name__=='__main__':main()