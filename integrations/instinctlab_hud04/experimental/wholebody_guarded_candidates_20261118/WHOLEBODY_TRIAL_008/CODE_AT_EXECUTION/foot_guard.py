"""Strict no-new-foot-frame guard from finite box/triangle SAT witnesses.
Box/triangle separating axes include all3box axes, triangle normal,9edgecross.
Thus negative strict SAT overlap is exact INTERSECTION detection, NOT depth.
Margin and support semantics are NOT accepted physical thresholds/labels.
"""
import numpy as np
def foot_state(body,gaps,records):
    frames=set();score=0.
    for g,r in zip(gaps,records):
        s=body.shapes[r['first']]
        if r['kind']!='terrain' or s.kind!='box' or s.link not in ['left_ankle_roll_link','right_ankle_roll_link']:continue
        score+=max(0.,-float(g))
        if g < -1e-7:frames.add((r['frame'],s.link))
    return frames,score
def guard(old,new):
    new_frames=new[0]-old[0];score_increase=new[1]-old[1]
    return not new_frames and score_increase<=1e-10,dict(new_foot_collision_frames=len(new_frames),
        old_foot_collision_frames=len(old[0]),new_total_foot_collision_frames=len(new[0]),
        old_foot_projection_overlap_score_raw=old[1],new_foot_projection_overlap_score_raw=new[1],
        footprint_score_not_signed_penetration_depth=True)