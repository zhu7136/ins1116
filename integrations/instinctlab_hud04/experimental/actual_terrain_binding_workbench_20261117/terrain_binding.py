"""Hash-bound ACTUAL terrain surface, explicit layout, no provenance guesses.

CPU software diagnostics only. Source capsule adapter never alters HUD04 cylinders.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

SOLID=Path('/workspace/hud04-solid-geometry-20261117')
INTEGRATION=Path('/workspace/hud04-recovery-bf4a057/integrations/instinctlab_hud04')
sys.path[:0]=[str(SOLID),str(INTEGRATION)]
from terrain_shape_query import TerrainShapeIndex
from robot_shape_inventory import check_rigid
from terrain_surface_geometry import processed_terrain,SurfaceIndex

TERRAIN_SHA='3aee33efea6793974d901dd0f7e46859e2d12d2dfd79f38f12c777686566c2de'
REVIEW_SHA='e14aef7350b2e957edb9d3aa4b6f83bcfe9051c17c0d544992c652a7e4d169ed'
DATA=Path('/workspace/hud04-original-drive-20261117')


def require_hash(path,digest):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Pinned input changed')


def translation(offset):
    offset=np.asarray(offset,float)
    if offset.shape!=(3,) or not np.isfinite(offset).all():raise ValueError('Explicit finite translation')
    t=np.eye(4);t[:3,3]=offset;return t


def layout_transform(size,border,*,row,col,num_rows,num_cols):
    size=np.asarray(size,float)
    if size.shape!=(2,) or not np.isfinite(size).all() or np.any(size<=0) or not np.isfinite(border):raise ValueError('Finite explicit layout required')
    if any(type(n) is not int for n in [row,col,num_rows,num_cols]) or not(1<=num_rows<=100 and 1<=num_cols<=100 and 0<=row<num_rows and 0<=col<num_cols):raise ValueError('Bounded valid tile index required')
    # Reviewed terrain function -> generator recenter -> tile -> layout center.
    pieces=[translation([size[0]/2,size[1]/2,-border]),translation([-size[0]/2,-size[1]/2,0]),
        translation([(row+.5)*size[0],(col+.5)*size[1],0]),translation([-num_rows*size[0]/2,-num_cols*size[1]/2,0])]
    transform=np.eye(4)
    for piece in pieces:transform=piece@transform
    direct=translation([(row+.5-num_rows/2)*size[0],(col+.5-num_cols/2)*size[1],-border])
    if not np.allclose(transform,direct,atol=1e-12,rtol=0):raise ValueError('Layout chain mismatch')
    return transform,pieces


class SourceCapsule:
    """Reviewed source G1 segment + radius; NEVER a target cylinder substitute."""
    def __init__(self,start,end,radius):
        self.endpoints=np.asarray([start,end],float);self.radius=float(radius)
        if self.endpoints.shape!=(2,3) or not np.isfinite(self.endpoints).all() or not np.isfinite(self.radius) or self.radius<=0:raise ValueError('Finite source capsule required')

    def support(self,directions,world):
        d=np.asarray(directions,float);world=check_rigid(world)
        if d.ndim!=2 or d.shape[1]!=3 or not 1<=len(d)<=256 or not np.isfinite(d).all():raise ValueError('Bounded finite directions required')
        ends=self.endpoints@world[:3,:3].T+world[:3,3]
        n=np.linalg.norm(d,axis=1)
        unit=np.divide(d,n[:,None],out=np.zeros_like(d),where=n[:,None]>0)
        return ends[np.argmax(ends@d.T,axis=0)]+self.radius*unit

    def aabb(self,world):
        p=self.support(np.vstack([np.eye(3),-np.eye(3)]),world)
        return np.diag(p[3:]),np.diag(p[:3])


class ActualTerrain:
    def __init__(self):
        self.path=DATA/'50cmboxes_19_36_49.stl';self.review_path=INTEGRATION/'PROCESSED_TERRAIN_SURFACES_20261117.json'
        require_hash(self.path,TERRAIN_SHA);require_hash(self.review_path,REVIEW_SHA)
        self.historical=json.loads(self.review_path.read_text())
        self.mesh,self.shift,self.stats=processed_terrain(self.path,[9.,12.])
        for key in ['input_faces','input_processed_vertices','cropped_faces','cropped_vertices','border_vertex_count','border_height_raw','terrain_function_shared_shift_raw']:
            if self.stats[key]!=self.historical['terrain'][key]:raise ValueError('Terrain preprocessing drift: '+key)
        self.triangles=np.array(self.mesh.triangles);self.normals=np.array(self.mesh.face_normals)
        self.lateral_ids=np.flatnonzero(np.abs(self.normals[:,2])<=.1)
        if len(self.triangles)!=8068 or len(self.lateral_ids)!=2958:raise ValueError('Surface inventory changed')
        self.function_transform=translation(self.shift)
        self.final_transform,self.chain=layout_transform([9.,12.],self.stats['border_height_raw'],row=0,col=0,num_rows=1,num_cols=1)

    def index(self,*,subset,transform):
        if subset not in ['all','lateral']:raise ValueError('Explicit surface subset required')
        ids=np.arange(len(self.triangles)) if subset=='all' else self.lateral_ids
        return TerrainShapeIndex(self.triangles[ids],terrain_to_world=transform),ids

    def verify_unchanged(self):
        require_hash(self.path,TERRAIN_SHA);require_hash(self.review_path,REVIEW_SHA)

    def evidence(self):
        return dict(input_sha256=TERRAIN_SHA,historical_report_sha256=REVIEW_SHA,preprocessing=self.stats,
            surface_faces=len(self.triangles),lateral_faces=len(self.lateral_ids),watertight=bool(self.mesh.is_watertight),
            triangles_float64_sha256=hashlib.sha256(self.triangles.astype('<f8').tobytes()).hexdigest(),
            terrain_function_transform=self.function_transform.tolist(),single_tile_final_transform=self.final_transform.tolist(),
            single_tile_is_explicit_software_layout_not_observed_runtime=True,
            no_caps_added=True,no_extra_perimeter_walls_added=True,units_and_source_alignment_proven=False)