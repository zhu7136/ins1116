"""Finite persisted full-sequence extension. No reference-acceptance writer.

Each window owns an exclusive intent, subprocess deadline, input checkpoint hash,
and optional output. `batch` never reruns a window with any existing intent.
Read-only final audit is separate so it has its own bounded validation budget.
"""
import argparse
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
from scipy.spatial.transform import Rotation
from material_problem import HERE, REPO, SEED_SHA, problem, perturb, hard_rows, feasibility, sha
from clearance_rows import build

OUT=HERE/'TRIAL_MATERIAL_003_FULL_SEQUENCE'
PROJECT='zhul136-university-of-science-and-technology-of-china/ins1116'
N=997


def put(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def windows():
    return [(int(s),int(s)+60) for s in np.rint(np.linspace(0,936,17)).astype(int)]


def load(p,names):
    with np.load(p,allow_pickle=False) as z:
        if z['joint_names'].tolist()!=names or z['joint_pos'].shape!=(N,29):raise ValueError('Named full997 state required')
        if z['frame_indices'].tolist()!=list(range(N)) or float(z['framerate'])!=90:raise ValueError('Native frame order/fps')
        if bool(z['accepted_reference']) or bool(z['training_ready']):raise ValueError('Never load an acceptance marker as an experimental seed')
        return z['joint_pos'].copy(),z['root_transforms_w'].copy()


def save(p,q,r,names):
    if Path(p).exists():raise FileExistsError(p)
    np.savez_compressed(p,joint_pos=q,joint_names=np.array(names),root_transforms_w=r,
                        frame_indices=np.arange(N),framerate=np.array(90),accepted_reference=np.array(False),
                        training_ready=np.array(False),artifact_role=np.array('FULL_LENGTH_ENGINEERING_CANDIDATE_NOT_TRAINING_INPUT'))


def root_smooth_rows(r,first,last):
    """Soft root-translation velocity and acceleration, incl. fixed neighbours.

    Weights are numerical regularizers, not invented physical acceptance limits.
    World-axis tangent rotations are still bounded and geometry guarded.
    """
    rows=[];cols=[];data=[];res=[];n=last-first+1
    for order,coeff,weight in [(1,[-1,1],.1),(2,[1,-2,1],1.)]:
        for start in range(max(0,first-order),min(N-order,last+1)):
            for c in range(3):
                row=len(res)
                res.append(weight*sum(v*r[start+k,c,3] for k,v in enumerate(coeff)))
                for k,v in enumerate(coeff):
                    f=start+k
                    if first<=f<=last:rows.append(row);cols.append(35*(f-first)+c);data.append(weight*v)
    return sparse.coo_matrix((data,(rows,cols)),shape=(len(res),n*35)).tocsr(),np.array(res)


def window_worker(index):
    p=OUT/f'WINDOW_{index:02d}';intent=json.loads((p/'INTENT.json').read_text())
    checkpoint=OUT/intent['input_checkpoint']
    if sha(checkpoint)!=intent['input_sha256']:raise ValueError('Window input changed')
    m,_,_,want,w,support,body,guard,kernel=problem();q,r=load(checkpoint,m.names)
    first,last=windows()[index];piece=q[first:last+1];roots=r[first:last+1];fs=slice(first,last+1)
    n=len(piece);nv=35*n;start=time.monotonic();deadline=start+55
    def phase(name):
        with (p/'PHASES.jsonl').open('a') as f:f.write(json.dumps(dict(phase=name,wall_s=time.monotonic()-start))+'\n')
        if time.monotonic()>deadline:raise TimeoutError('Window computation deadline before '+name)
    report=dict(index=index,first=first,last=last,lp_calls=0,accepted_update=False,accepted_reference=False,
                training_ready=False,gpu_submissions=0,ppo_updates=0,backtracking=[])
    value,j=m.evaluate(piece,roots,True)
    a=sparse.block_diag([sparse.csr_matrix(v*x[:,None]) for v,x in zip(j,w[fs])],format='csr')
    res=((value-want[fs])*w[fs]).ravel()
    sm,sr=root_smooth_rows(r,first,last);a=sparse.vstack([a,sm],format='csr');res=np.r_[res,sr];na=len(res)
    hard,rhs=hard_rows(m,q,first,last);clear,crhs,_=build(body,kernel,m,piece,roots);phase('hard_clearance_rows')
    gaps,gj,records,counts=body.evaluate(m,piece,roots,True);phase('geometry_linearization')
    same=np.array([x['kind']=='self' and body.shapes[x['first']].link==body.shapes[x['second']].link for x in records],bool)
    movable=np.asarray(gj.getnnz(axis=1)>0).ravel() & ~same
    gl,jl=gaps[movable],gj[movable];ng=len(gl)
    mat=sparse.vstack([
        sparse.hstack([hard,sparse.csr_matrix((len(rhs),na+ng))]),
        sparse.hstack([clear,sparse.csr_matrix((len(crhs),na+ng))]),
        sparse.hstack([a,-sparse.eye(na),sparse.csr_matrix((na,ng))]),
        sparse.hstack([-a,-sparse.eye(na),sparse.csr_matrix((na,ng))]),
        sparse.hstack([-1000*jl,sparse.csr_matrix((ng,na)),-sparse.eye(ng)])],format='csr')
    b=np.r_[rhs,crhs,-res,res,1000*(gl-body.margin)]
    trust=np.tile(np.r_[np.full(3,.015),np.full(3,.04),np.full(29,.08)],n)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=OptimizeWarning)
        lp=linprog(np.r_[np.zeros(nv),np.ones(na+ng)],A_ub=mat,b_ub=b,
                   bounds=np.c_[-trust,trust].tolist()+[(0,None)]*(na+ng),method='highs',
                   options={'time_limit':3.,'maxiter':15000,'threads':2,'parallel':False})
    report.update(lp_calls=1,lp_status=int(lp.status),lp_message=lp.message,hard_clearance_rows=len(crhs));phase('LP_complete')
    if lp.success:
        step=lp.x[:nv].reshape(n,35);old=float(abs(res).sum()+1000*np.maximum(body.margin-gaps,0).sum())
        old_state=guard.scan(m,piece,roots,np.arange(first,last+1));phase('old_endpoint_certificates')
        # Prior probe showed alpha1/.25 regressing certificates. Start .0625,
        # then shrink twice; nonlinear verification still mandatory each time.
        for alpha in [.0625,.015625,.00390625]:
            phase('before_candidate')
            tq,tr=piece+alpha*step[:,6:],perturb(roots,alpha*step[:,:6])
            if np.max(hard@(alpha*step.ravel())-rhs)>1e-6 or np.max(clear@(alpha*step.ravel())-crhs)>1e-6:
                raise ValueError('Linear original-limit/clearance contract violated')
            nr=r.copy();nr[fs]=tr
            _,nsr=root_smooth_rows(nr,first,last)
            ngaps,_,_=body.evaluate(m,tq,tr,False);phase('fresh_candidate_geometry')
            merit=float(abs((m.evaluate(tq,tr)-want[fs])*w[fs]).sum()+abs(nsr).sum()+1000*np.maximum(body.margin-ngaps,0).sum())
            detail=dict(alpha=alpha,old_merit=old,merit=merit,improved=merit<old-1e-9)
            report['backtracking'].append(detail)
            if not detail['improved']:continue
            state=guard.scan(m,tq,tr,np.arange(first,last+1));phase('candidate_endpoint_certificates')
            passed,check=guard.compare(old_state,state);detail['endpoint_guard']=check
            if not passed:continue
            q[fs],r[fs]=tq,tr
            if max(feasibility(m,q).values())>1e-6:raise ValueError('Whole-sequence original position/velocity violation')
            save(p/'CANDIDATE_UNACCEPTED.npz',q,r,m.names)
            report.update(accepted_update=True,alpha=alpha,candidate_sha256=sha(p/'CANDIDATE_UNACCEPTED.npz'))
            break
    guard.verify_inputs();report['wall_s']=time.monotonic()-start;put(p/'RESULT.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='backtracking'}),flush=True)


def prepare():
    tests=Path('/workspace/hud04-full003-tests.log').read_text()
    if 'FAILED' in tests or not tests.rstrip().endswith('OK'):raise ValueError('Pretests must pass')
    OUT.mkdir(exist_ok=False)
    m,q,r,_,_,_,_,_,_=problem()
    partial=HERE/'TRIAL_MATERIAL_002_HARD_CLEARANCE_PROBE/WINDOW0_14_UNACCEPTED.npz'
    if sha(partial)!='44590c5971dc1a4a0e6cd44ffb354001b18b055846c0a24e6301214625419ad0':raise ValueError('Verified warm start changed')
    with np.load(partial,allow_pickle=False) as z:q[:15],r[:15]=z['joint_pos'],z['root_transforms_w']
    if max(feasibility(m,q).values())>1e-6:raise ValueError('Warm-start feasibility')
    save(OUT/'INITIAL_UNACCEPTED.npz',q,r,m.names)
    config=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                authority='2026-11-18T05:31:52Z continue toward acceptable motion; bounded CPU pass',
                source_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
                warm_start_sha256=sha(partial),seed_sha256=SEED_SHA,frames=N,fps=90,body_dof=29,head_dof=0,
                windows=windows(),maximum_lp_calls=17,maximum_endpoint_candidates_per_window=3,
                window_soft_deadline_s=55,window_supervisor_s=65,total_window_supervisor_cap_s=1105,
                final_audit_separate_supervisor_s=100,automatic_retries=0,accepted_reference=False,
                training_ready=False,gpu_submissions=0,ppo_updates=0,wandb_run_id=uuid.uuid4().hex[:8])
    provenance={k:os.environ[e] for k,e in [('thread_id','WANDB_ARIA_THREAD_ID'),('turn_id','WANDB_ARIA_TURN_ID')] if e in os.environ}
    if provenance:config['_wb_agent']=provenance
    put(OUT/'INTENT.json',config)
    snap=OUT/'CODE_AT_EXECUTION';snap.mkdir()
    for f in HERE.glob('*.py'):(snap/f.name).write_bytes(f.read_bytes())
    put(OUT/'CODE_SHA256.json',{f.name:sha(f) for f in snap.iterdir()})
    import wandb
    run=wandb.init(entity=PROJECT.split('/')[0],project='ins1116',id=config['wandb_run_id'],config=config,
                    name='hud04-material-full-003-NOT-ACCEPTED',job_type='cpu-material-full-sequence',
                    dir=str(OUT),settings=wandb.Settings(disable_git=True,save_code=False))
    run.summary.update(dict(computation_complete=False,accepted_reference=False,training_ready=False,lp_calls=0,gpu_submissions=0,ppo_updates=0))
    # Bookkeeping session exits; computation is synchronous per bounded window.
    run.finish();print(json.dumps(config),flush=True)


def batch(count):
    intent=json.loads((OUT/'INTENT.json').read_text());checkpoint=OUT/'INITIAL_UNACCEPTED.npz';done=0
    for i in range(17):
        p=OUT/f'WINDOW_{i:02d}'
        if p.exists():
            if not (p/'SUPERVISION.json').exists():raise ValueError('Unfinished intent: inspect, NEVER duplicate')
            previous=json.loads((p/'SUPERVISION.json').read_text())
            if previous['child_exit']!=0 or previous['timed_out'] or not previous['child_reaped']:
                raise ValueError('Failed prior window: no automatic retry OR continuation')
            wi=json.loads((p/'INTENT.json').read_text())
            if wi['input_checkpoint']!=str(checkpoint.relative_to(OUT)) or wi['input_sha256']!=sha(checkpoint):
                raise ValueError('Prior checkpoint chain mismatch')
            if (p/'CANDIDATE_UNACCEPTED.npz').exists():
                row=json.loads((p/'RESULT.json').read_text())
                if not row['accepted_update'] or sha(p/'CANDIDATE_UNACCEPTED.npz')!=row['candidate_sha256']:raise ValueError('Invalid checkpoint')
                checkpoint=p/'CANDIDATE_UNACCEPTED.npz'
            continue
        if done>=count:break
        p.mkdir(exist_ok=False)
        put(p/'INTENT.json',dict(index=i,input_checkpoint=str(checkpoint.relative_to(OUT)),input_sha256=sha(checkpoint),
                                automatic_retries=0,supervisor_s=65,source_commit=intent['source_commit']))
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
        start=time.monotonic();timedout=False
        with (p/'EXECUTION.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__)),'worker','--index',str(i)],env=env,stdout=log,stderr=subprocess.STDOUT)
            try:child.wait(timeout=65)
            except subprocess.TimeoutExpired:
                timedout=True;child.terminate()
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2)
        receipt=dict(index=i,child_exit=child.returncode,child_reaped=child.poll() is not None,timed_out=timedout,wall_s=time.monotonic()-start)
        put(p/'SUPERVISION.json',receipt)
        if child.returncode!=0:
            print(json.dumps(receipt),flush=True)
            raise RuntimeError('One window failed: inspect and change approach, no automatic retry/continuation')
        report=json.loads((p/'RESULT.json').read_text());print(json.dumps(dict(index=i,first=report['first'],last=report['last'],accepted=report['accepted_update'],wall_s=report['wall_s'])),flush=True)
        if report['accepted_update']:checkpoint=p/'CANDIDATE_UNACCEPTED.npz'
        done+=1


def audit_worker():
    m,_,_,want,w,support,body,guard,kernel=problem();checkpoint=OUT/'INITIAL_UNACCEPTED.npz';rows=[]
    for i in range(17):
        p=OUT/f'WINDOW_{i:02d}';sup=json.loads((p/'SUPERVISION.json').read_text())
        if sup['child_exit']!=0 or not sup['child_reaped']:raise ValueError('All windows must be terminal and successful')
        row=json.loads((p/'RESULT.json').read_text());rows.append(row)
        if row['accepted_update']:checkpoint=p/'CANDIDATE_UNACCEPTED.npz'
    iq,ir=load(OUT/'INITIAL_UNACCEPTED.npz',m.names);q,r=load(checkpoint,m.names)
    start=time.monotonic();old=guard.scan(m,iq,ir,np.arange(N));new=guard.scan(m,q,r,np.arange(N));passed,check=guard.compare(old,new)
    if not passed:raise ValueError('Whole997 nonlinear geometry regression')
    lim=feasibility(m,q)
    if max(lim.values())>1e-6:raise ValueError('Original limits failed')
    save(OUT/'FULL997_ENGINEERING_CANDIDATE_UNACCEPTED.npz',q,r,m.names)
    # Also export canonical reader layout, while explicitly rejecting training use.
    quat=Rotation.from_matrix(r[:,:3,:3]).as_quat()[:,[3,0,1,2]]
    for i in range(1,N):
        if quat[i]@quat[i-1]<0:quat[i]*=-1
    np.savez_compressed(OUT/'STEP1_READER_LAYOUT_UNACCEPTED.npz',joint_pos=q,joint_names=np.array(m.names),
                        base_pos_w=r[:,:3,3],base_quat_w=quat,framerate=np.array(90),frame_indices=np.arange(N),
                        accepted_reference=np.array(False),training_ready=np.array(False),
                        artifact_role=np.array('REJECTED_ENGINEERING_CANDIDATE_NOT_APPROVED_REFERENCE'))
    def stats(q,r):
        v=m.evaluate(q,r);err=np.linalg.norm((v[:,:18]-want[:,:18]).reshape(N,6,3),axis=2)
        return dict(material_position_mean_m=float(err.mean()),material_position_max_m=float(err.max()),
                    proposed_support_position_mean_m=float(err[support].mean()),proposed_support_position_max_m=float(err[support].max()),
                    root_speed_max_m_s=float(np.linalg.norm(np.diff(r[:,:3,3],axis=0)*90,axis=1).max()),
                    root_acceleration_max_m_s2=float(np.linalg.norm(np.diff(r[:,:3,3],n=2,axis=0)*8100,axis=1).max()),
                    joint_acceleration_max_rad_s2=float(abs(np.diff(q,n=2,axis=0)*8100).max()))
    self_counts={}
    for kind,state in [('initial',old),('final',new)]:
        from collections import Counter
        c=Counter((k[2],k[3]) for k in state.strict if k[0]=='self')
        self_counts[kind]=[dict(first=guard.inventory[a][1].identity,second=guard.inventory[b][1].identity,frame_count=count) for (a,b),count in c.most_common()]
    report=dict(covered_frames=997,visited_windows=17,lp_calls=sum(x['lp_calls'] for x in rows),accepted_updates=sum(x['accepted_update'] for x in rows),
                complete_single_pass=True,convergence_established=False,full_endpoint_guard=check,original_position_velocity_limits=lim,
                initial=stats(iq,ir),final=stats(q,r),self_strict_by_pair=self_counts,
                unassigned_support_frames=346,contact_or_flight_inferred=False,physical_effort_verified=False,
                swept_motion_verified=False,actual_runtime_filter_policy_verified=False,accepted_reference=False,training_ready=False,
                full_candidate_exported=True,approved_reference_exported=False,gpu_submissions=0,ppo_updates=0,
                final_audit_wall_s=time.monotonic()-start,candidate_sha256=sha(OUT/'FULL997_ENGINEERING_CANDIDATE_UNACCEPTED.npz'))
    put(OUT/'RESULT.json',report);guard.verify_inputs()
    intent=json.loads((OUT/'INTENT.json').read_text())
    import wandb
    run=wandb.init(entity=PROJECT.split('/')[0],project='ins1116',id=intent['wandb_run_id'],resume='must',dir=str(OUT),settings=wandb.Settings(disable_git=True,save_code=False))
    run.summary.update({k:v for k,v in report.items() if isinstance(v,(int,float,bool,str))})
    run.summary.update(dict(newly_strict_pairs=check['newly_strict_pairs'],lost_certified_separation_pairs=check['lost_certified_separation_pairs'],
                            strict_pair_frame_records=check['new_strict_pairs'],unresolved_pair_frame_records=check['new_unresolved_pairs'],computation_complete=True))
    run.finish();print(json.dumps({k:v for k,v in report.items() if k!='self_strict_by_pair'},indent=2),flush=True)


def audit():
    p=OUT/'AUDIT_INTENT.json';put(p,dict(supervisor_s=100,automatic_retries=0))
    with (OUT/'AUDIT_EXECUTION.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__)),'audit-worker'],stdout=log,stderr=subprocess.STDOUT)
        timedout=False
        try:child.wait(timeout=100)
        except subprocess.TimeoutExpired:
            timedout=True;child.terminate()
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2)
    receipt=dict(child_exit=child.returncode,child_reaped=child.poll() is not None,timed_out=timedout,result_exists=(OUT/'RESULT.json').exists())
    put(OUT/'AUDIT_SUPERVISION.json',receipt);print(json.dumps(receipt),flush=True)
    if child.returncode:raise RuntimeError('Independent audit failed; no retry')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','batch','worker','audit','audit-worker']);p.add_argument('--count',type=int,default=1);p.add_argument('--index',type=int);args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='batch':
        if not 1<=args.count<=2:raise ValueError('At most2 windows per synchronous controller batch')
        batch(args.count)
    elif args.action=='worker':window_worker(args.index)
    elif args.action=='audit':audit()
    else:audit_worker()