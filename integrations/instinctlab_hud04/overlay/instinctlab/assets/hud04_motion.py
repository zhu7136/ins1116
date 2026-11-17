"""CPU-only strict Step1 loading and reviewed terrain/support metadata."""
import hashlib,json
from pathlib import Path
import zipfile
import numpy as np

SUPPORT_LINKS=('left_ankle_roll_link','right_ankle_roll_link','left_wrist_roll_link',
               'right_wrist_roll_link','left_knee_link','right_knee_link')

def permutation(stored,runtime):
    stored,runtime=list(stored),list(runtime)
    if len(stored)!=29 or len(runtime)!=29 or len(set(stored))!=29 or len(set(runtime))!=29 or set(stored)!=set(runtime) or any('head' in n.lower() for n in stored):
        raise ValueError('Exactly the same 29 unique named body DOFs required')
    return [stored.index(n) for n in runtime]

def read_named_motion(path,runtime_names,limits):
    path=Path(path)
    if not path.name.endswith(('retargeted.npz','retargetted.npz')) or path.stat().st_size>128*1024*1024:
        raise ValueError('Reviewed NPZ only; no joint-fit diagnostic files')
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist()
        if len({x.filename for x in entries})!=len(entries) or sum(x.file_size for x in entries)>256*1024*1024:
            raise ValueError('Duplicate or oversized NPZ members')
    with np.load(path,allow_pickle=False) as z:
        names=z['joint_names']
        if names.ndim!=1 or names.dtype.kind!='U':raise ValueError('Unicode names only; no pickles')
        order=permutation(names.tolist(),runtime_names)
        q=np.asarray(z['joint_pos'],float);p=np.asarray(z['base_pos_w'],float);r=np.asarray(z['base_quat_w'],float)
        fps=np.asarray(z['framerate'])
        if fps.size!=1 or float(fps.item())!=90 or q.shape!=(997,29) or p.shape!=(997,3) or r.shape!=(997,4):
            raise ValueError('Full 997-frame native 90Hz Step1 required')
        if not all(np.isfinite(a).all() for a in (q,p,r)) or not np.allclose(np.linalg.norm(r,axis=1),1,atol=1e-3,rtol=0):
            raise ValueError('Finite arrays and unit wxyz quaternions required; never normalize/infer')
        q=q[:,order].copy()
        lower,upper,vel=[np.array([limits[n][k] for n in runtime_names]) for k in ('lower','upper','velocity')]
        if np.any(q<lower-1e-6) or np.any(q>upper+1e-6) or np.any(np.abs(np.diff(q,axis=0))*90>vel+1e-6):
            raise ValueError('Model position/velocity limits violated; never clip')
        return q,p.copy(),r.copy(),90.

def _file(root,name,digest):
    p=(root/name).resolve()
    if Path(name).is_absolute() or not p.is_relative_to(root) or not p.is_file() or p.stat().st_size>128*1024*1024:
        raise ValueError('Unsafe reviewed file path/size')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Reviewed file checksum')
    return p

def validate_support(document,motion_files):
    if document.get('schema_version')!=1 or document.get('status')!='reviewed_support_windows' or document.get('framerate')!=90 or document.get('frames')!=997:
        raise ValueError('Reviewed support schema required; no guessed support windows')
    motions=document.get('motions',{})
    if set(motions)!=set(motion_files):raise ValueError('Support/motion identity mismatch')
    for links in motions.values():
        if not isinstance(links,dict) or set(links)!=set(SUPPORT_LINKS):raise ValueError('All six support-link schedules explicitly required; [] means never')
        for windows in links.values():
            if not isinstance(windows,list):raise ValueError('Window list required')
            end=-1
            for window in windows:
                if not isinstance(window,list) or len(window)!=2 or any(type(x)is not int for x in window):raise ValueError('Integer inclusive intervals required')
                a,b=window
                if not end<a<=b<997:raise ValueError('Ordered disjoint 0-based support intervals required')
                end=b
    return document

def validate_terrain_bundle(manifest_path):
    import yaml
    path=Path(manifest_path).resolve();root=path.parent
    m=json.loads(path.read_text())
    if m.get('status')!='accepted_for_terrain_replay':raise ValueError('Terrain replay manifest required')
    for k in ('terrain','source_conventions','support_intervals'):
        if not isinstance(m.get('review_evidence',{}).get(k),str) or not m['review_evidence'][k].strip():raise ValueError('Missing '+k+' review')
    meta=m['terrain_metadata'];metadata=_file(root,meta['path'],meta['sha256'])
    y=yaml.safe_load(metadata.read_text())
    motions=y.get('motion_files',[]);terrains=y.get('terrains',[])
    # Deliberately Step1-only. Current upstream match_scene indexes terrain IDs
    # numerically; do not accept generic/string identifiers it cannot honor.
    expected=[f['path'] for f in m['files']]
    if len(motions)!=1 or [x.get('motion_file') for x in motions]!=expected or type(motions[0].get('terrain_id')) is not int or motions[0]['terrain_id']!=0:
        raise ValueError('Exactly reviewed Step1 motion with integer terrain_id 0 required')
    if len(terrains)!=1 or type(terrains[0].get('terrain_id')) is not int or terrains[0]['terrain_id']!=0:raise ValueError('Exactly terrain 0 required')
    mesh=terrains[0]['terrain_file']
    if mesh!=m['terrain_mesh']['path']:raise ValueError('Terrain file identity mismatch')
    _file(root,mesh,m['terrain_mesh']['sha256'])
    support=m['support_schedule'];p=_file(root,support['path'],support['sha256'])
    schedule=validate_support(json.loads(p.read_text()),expected)
    return root,metadata,schedule

def allowed_support_mask(body_names,source_frames,links):
    """CPU reference implementation; no force/contact inference or global whitelist."""
    frames=np.asarray(source_frames)
    if frames.ndim!=1 or frames.dtype.kind not in 'iu':raise ValueError('Integer source-frame vector required')
    mask=np.zeros((len(frames),len(body_names)),dtype=bool)
    for col,name in enumerate(body_names):
        for a,b in links.get(name,[]):mask[:,col]|=(frames>=a)&(frames<=b)&(frames>=0)&(frames<997)
    return mask