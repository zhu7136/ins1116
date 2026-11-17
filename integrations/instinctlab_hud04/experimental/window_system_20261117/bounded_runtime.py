"""Trusted CPU worker supervision, not an arbitrary-code security sandbox.

No motion writer or accepted-manifest writer exists here. Results are bounded
JSON diagnostics. Timeout uses our own child handle, never process discovery.
The iteration gate MUST be called by each trusted worker/solver iteration;
opaque solvers without a compatible callback are not supported by this API.
"""
from dataclasses import dataclass, asdict
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time

MAX_MESSAGE = 32768
THREAD_ENV = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')


@dataclass(frozen=True)
class Budget:
    window_seconds: float = 90
    total_seconds: float = 600
    iterations: int = 100
    numeric_threads: int = 2
    max_windows: int = 3

    def __post_init__(self):
        for name, maximum in [('window_seconds',90),('total_seconds',600)]:
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<value<=maximum:
                raise ValueError('Invalid bounded '+name)
        for name,maximum in [('iterations',100),('numeric_threads',2),('max_windows',3)]:
            value=getattr(self,name)
            if type(value) is not int or not 0<value<=maximum:raise ValueError('Invalid bounded '+name)


class IterationLimit(RuntimeError):pass


class IterationGate:
    def __init__(self,limit):self.limit=limit;self.count=0
    def step(self):
        if self.count>=self.limit:raise IterationLimit('iteration_budget_exhausted')
        self.count+=1


def _json_bytes(value):
    raw=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
    if len(raw)>MAX_MESSAGE:raise ValueError('diagnostic_message_size_limit')
    return raw


def _child(connection,module_name,function_name,payload,budget):
    gate=IterationGate(budget.iterations)
    try:
        # This child starts with spawn. Set numerical-library thread preferences
        # before importing the trusted worker, then verify supported pools too.
        for name in THREAD_ENV:os.environ[name]=str(budget.numeric_threads)
        os.environ['OMP_DYNAMIC']='FALSE'
        os.environ['CUDA_VISIBLE_DEVICES']=''
        from threadpoolctl import threadpool_info,threadpool_limits
        module=importlib.import_module(module_name)
        function=getattr(module,function_name)
        with threadpool_limits(limits=budget.numeric_threads):
            result=function(gate,payload)
            pools=threadpool_info()
            if any(p['num_threads']>budget.numeric_threads for p in pools):raise RuntimeError('numeric_thread_limit_not_applied')
            message=dict(status='completed',iterations=gate.count,diagnostic=result,
                         supported_numeric_pools=[dict(api=p['internal_api'],threads=p['num_threads']) for p in pools])
        connection.send_bytes(_json_bytes(message))
    except BaseException as exc:
        # Do not forward arbitrary exception text: it may contain file contents
        # or unexpected large diagnostics. Error class plus gate count is enough.
        status='iteration_limit' if isinstance(exc,IterationLimit) else 'worker_error'
        try:connection.send_bytes(_json_bytes(dict(status=status,error_type=type(exc).__name__,iterations=gate.count)))
        except (OSError,ValueError):pass
    finally:connection.close()


def _finish_own_child(process):
    """Bounded cleanup of only the process created by this supervisor."""
    if process.is_alive():process.terminate()
    process.join(.2)
    if process.is_alive():process.kill();process.join(.5)
    return not process.is_alive()


def supervise(module_name,function_name,payloads,budget=Budget()):
    if not isinstance(budget,Budget):raise ValueError('Budget required')
    if not isinstance(payloads,list) or not 1<=len(payloads)<=budget.max_windows:raise ValueError('Bounded nonempty windows required')
    for payload in payloads:_json_bytes(payload)
    # Library path/function is trusted application code, NOT untrusted user code.
    start=time.monotonic();deadline=start+budget.total_seconds;windows=[]
    context=mp.get_context('spawn')
    for index,payload in enumerate(payloads):
        if time.monotonic()>=deadline:
            windows.append(dict(index=index,status='not_started_total_deadline'));break
        window_start=time.monotonic();window_deadline=min(deadline,window_start+budget.window_seconds)
        recv,send=context.Pipe(duplex=False)
        process=context.Process(target=_child,args=(send,module_name,function_name,payload,budget),daemon=True)
        try:
            process.start();send.close()
            message=None
            while time.monotonic()<window_deadline:
                remaining=window_deadline-time.monotonic()
                if recv.poll(min(.01,max(0,remaining))):
                    try:message=json.loads(recv.recv_bytes(MAX_MESSAGE))
                    except (EOFError,OSError,ValueError):message=dict(status='protocol_error')
                    break
                if not process.is_alive():
                    message=dict(status='worker_exit_without_result');break
            if message is None:message=dict(status='timeout')
            # Returning a message is not proof of child exit. A worker that
            # leaves a live non-daemon thread cannot produce a successful row.
            if message['status']=='completed':
                process.join(max(0,min(.05,window_deadline-time.monotonic())))
                while process.is_alive() and time.monotonic()<window_deadline:
                    process.join(min(.01,max(0,window_deadline-time.monotonic())))
                if process.is_alive():message=dict(status='timeout_after_result')
                elif process.exitcode!=0:message=dict(status='worker_nonzero_exit')
            clean=_finish_own_child(process)
            if not clean:message=dict(status='child_cleanup_failed')
            if message['status']=='completed' and time.monotonic()>window_deadline:message=dict(status='deadline_exceeded_at_completion')
            windows.append(dict(index=index,wall_seconds=time.monotonic()-window_start,child_reaped=clean,**message))
        finally:
            send.close();recv.close()
            if process.pid is not None:
                if process.is_alive():_finish_own_child(process)
                if not process.is_alive():process.close()
        # Failure stops the whole sequence; no implicit continuation or retry.
        if windows[-1]['status']!='completed':break
    completed=len(windows)==len(payloads) and all(w['status']=='completed' for w in windows)
    return dict(schema_version=1,stage='bounded_cpu_diagnostic_supervision',status='completed' if completed else 'stopped',
                budget=asdict(budget),windows=windows,wall_seconds=time.monotonic()-start,retries=0,
                hard_parent_deadline=True,cleanup_grace_seconds_max=.7,
                iteration_limit_scope='trusted_workers_calling_gate_each_iteration_only',
                thread_limit_scope='supported_BLAS_OpenMP_pools_not_all_OS_threads_or_CPU_cores',
                gpu_submissions=0,solver_calls=0,candidate_motion_generated=False,accepted_reference=False,training_ready=False)


def save_diagnostic(path,report):
    """Audit reports only, including failure; never a trajectory export."""
    path=Path(path)
    if path.suffix!='.json' or report.get('stage')!='bounded_cpu_diagnostic_supervision':raise ValueError('Diagnostic JSON only')
    if any(report.get(k) is not False for k in ['candidate_motion_generated','accepted_reference','training_ready']):raise ValueError('Acceptance or trajectory export forbidden')
    text=json.dumps(report,indent=2,allow_nan=False)
    with path.open('x') as f:f.write(text+'\n')
    if json.loads(path.read_text())!=report:raise RuntimeError('Diagnostic readback mismatch')