"""Single CPU counterexample reproduction; no remote services or file overwrite."""
import json
from pathlib import Path
import numpy as np
from terrain_binding import ActualTerrain
from robot_shape_inventory import load_shapes,TargetTasks
from independent_cylinder_refined import refined_bracket

if __name__=='__main__':
    terrain=ActualTerrain();tree,shapes,_=load_shapes();target=TargetTasks()
    q=np.array([(target.limits[n]['lower']+target.limits[n]['upper'])/2 for n in target.names]);world={n:p[0] for n,p in tree.fk(q[None],target.names).items()}
    shape=next(s for s in shapes if s.identity=='left_hip_pitch_link#0')
    result=refined_bracket(shape,world[shape.link],terrain.triangles)
    print(json.dumps(result,indent=2),flush=True)