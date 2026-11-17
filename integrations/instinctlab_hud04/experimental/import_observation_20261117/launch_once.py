"""NEW gm-run entry: locate this checkout, execute one bounded diagnostic only."""
import argparse,json,os,re,sys
from pathlib import Path
from types import SimpleNamespace
from worker import main as observe

def safe_tail(path):
    if not path.is_file():return ''
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-24576));text=f.read(24576).decode('utf-8',errors='replace')
    for name in ('github1117','flux1117','WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY'):
        value=os.environ.get(name)
        if value:text=text.replace(value,'[REDACTED]')
    text=re.sub(r'https?://\S+','[URL omitted]',text)
    text=re.sub(r'\b(?:gh[pousr]_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+|gm_sk_[A-Za-z0-9_]+)\b','[REDACTED]',text)
    text=re.sub(r'(?i)(api[-_ ]?key|authorization|password|secret|access[-_]?token)\s*[:=]\s*[^\s,]+',r'\1=[REDACTED]',text)
    return '\n'.join(text.splitlines()[-60:])[-8192:]

def safe_records(path):
    """Preserve bounded phase/original-error records even after long cleanup logs."""
    if not path.is_file():return []
    if path.stat().st_size>8*1024*1024:raise ValueError('Unexpected diagnostic log size')
    records=[]
    for line in path.read_text(errors='replace').splitlines():
        if not line.startswith('{"stage":'):continue
        try:row=json.loads(line)
        except ValueError:continue
        if row.get('stage') not in ('hud04_import_progress','hud04_probe_exception'):continue
        text=json.dumps(row)
        for name in ('github1117','flux1117','WANDB_API_KEY','OPENAI_API_KEY','FLUX_API_KEY'):
            value=os.environ.get(name)
            if value:text=text.replace(value,'[REDACTED]')
        text=re.sub(r'https?://[^\s"\\]+','[URL omitted]',text)
        text=re.sub(r'\b(?:gh[pousr]_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+|gm_sk_[A-Za-z0-9_]+)\b','[REDACTED]',text)
        try:row=json.loads(text)
        except ValueError:continue
        if len(text)<=10000:records.append(row)
        if len(records)>=32:break
    return records

def main(args):
    if not args.execute:
        observe(SimpleNamespace(execute=False));return
    if not re.fullmatch('[0-9a-f]{40}',args.expected_commit or ''):raise ValueError('Full expected commit required')
    repo=Path(__file__).resolve().parents[4]
    if not (repo/'flux/hud04_asset_camera_probe/manifest.json').is_file():raise ValueError('Expected checkout layout absent')
    output=repo/('hud04_import_observation_'+args.expected_commit[:12])
    print(json.dumps(dict(stage='hud04_import_entry_started',expected_commit=args.expected_commit,worker_seconds=240,
        automatic_retries=0,solver_calls=0,ppo_updates=0,accepted_reference=False,training_ready=False)),flush=True)
    try:
        observe(SimpleNamespace(execute=True,repo_root=str(repo),expected_commit=args.expected_commit,output=str(output),isaac_launcher=None))
    except BaseException as exc:
        print(json.dumps(dict(stage='hud04_import_entry_failed',error_type=type(exc).__name__,
            diagnostic_tail=safe_tail(output/'child.log'),diagnostic_records=safe_records(output/'child.log'),
            accepted_reference=False,training_ready=False)),flush=True)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--expected-commit');main(p.parse_args())