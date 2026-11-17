"""Phase-conditional support filtering. Current forces, not mixed-frame history.

Runtime integration is pinned to the inspected InstinctLab manager/buffer API;
CPU tests cannot establish physical contact correctness.
"""
from pathlib import Path
import torch
from instinctlab.assets.hud04_motion import allowed_support_mask

def _violations(env,threshold,sensor_cfg):
    manager=env.scene['motion_reference']
    if set(manager._motion_buffers)!={'TerrainMotion'}:raise RuntimeError('Single TerrainMotion buffer required')
    buffer=manager._motion_buffers['TerrainMotion']
    ids=torch.arange(env.num_envs,device=env.device)
    if not buffer.env_ids_is_assigned(ids).all():raise RuntimeError('Incomplete support buffer assignments')
    frames=buffer.support_source_frames(ids,manager._timestamp).detach().cpu().numpy()
    files=buffer.get_current_motion_identifiers(ids)
    sensor=env.scene.sensors[sensor_cfg.name]
    names=sensor.body_names
    masks=[]
    for i,file in enumerate(files):
        relative=str(Path(file).resolve().relative_to(Path(buffer.cfg.path).resolve()))
        # No fallback to feet-only or a global hand/knee whitelist.
        links=env.cfg.hud04_support_schedule['motions'][relative]
        masks.append(allowed_support_mask(names,frames[i:i+1],links)[0])
    import numpy as np
    allowed=torch.as_tensor(np.stack(masks),device=env.device,dtype=torch.bool)
    force=torch.linalg.vector_norm(sensor.data.net_forces_w,dim=-1)
    if allowed.shape!=force.shape:raise RuntimeError('Contact sensor/body mapping mismatch')
    return (force>threshold)&~allowed

def conditional_undesired_contacts(env,threshold,sensor_cfg):
    return _violations(env,threshold,sensor_cfg).sum(dim=-1).float()

def illegal_support_contact(env,threshold,sensor_cfg):
    return _violations(env,threshold,sensor_cfg).any(dim=-1)