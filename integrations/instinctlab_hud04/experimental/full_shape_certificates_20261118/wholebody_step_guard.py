"""Fail-closed endpoint classification guard, NOT physical motion acceptance.

Every distinct-link pair and every shape/finite-terrain broadphase is refreshed.
Previously certified separated pairs may not become strict OR unresolved. New
strict witnesses are rejected even for a previously unresolved pair. Existing
intersections are not declared safe, and witness margins are NOT depth scores.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from audit_self_pairs import load_case, TARGET, HERE, sha
from nominal_certificates import classify, classify_triangles, rigid, finite


@dataclass(frozen=True)
class EndpointState:
    binding: str
    frames: tuple
    strict: frozenset
    unresolved: frozenset
    self_records: int
    terrain_records: int


def compare_states(old, new):
    if not isinstance(old, EndpointState) or not isinstance(new, EndpointState):
        raise ValueError('Explicit guarded endpoint states required')
    if old.binding != new.binding or old.frames != new.frames:
        raise ValueError('Geometry/frame binding mismatch')
    if (old.self_records, old.terrain_records) != (new.self_records, new.terrain_records):
        raise ValueError('Incomplete endpoint accounting')
    for state in [old, new]:
        if not isinstance(state.binding,str) or len(state.binding)!=64 or any(c not in '0123456789abcdef' for c in state.binding):
            raise ValueError('Explicit geometry digest required')
        if state.strict & state.unresolved:
            raise ValueError('Contradictory classifications')
        if not state.frames or len(set(state.frames)) != len(state.frames):
            raise ValueError('Nonempty unique frames required')
        if any(type(f) is not int or not 0<=f<997 for f in state.frames) or tuple(sorted(state.frames))!=state.frames:
            raise ValueError('Ordered native integer frames required')
        if state.self_records!=len(state.frames)*903 or state.terrain_records!=len(state.frames)*43*8068:
            raise ValueError('All43/all903 accounting required')
        for k in state.strict | state.unresolved:
            if not isinstance(k,tuple) or len(k)!=4 or k[0] not in ('self','terrain') or any(type(x) is not int for x in k[1:]):
                raise ValueError('Explicit classified pair identity required')
            if not 0<=k[2]<43 or not 0<=k[3]<(43 if k[0]=='self' else 8068) or (k[0]=='self' and k[2]>=k[3]):
                raise ValueError('Pair identity outside inventory')
        if any(k[1] not in state.frames for k in state.strict | state.unresolved):
            raise ValueError('Witness outside bound frames')
    lost_separation = (new.strict | new.unresolved) - (old.strict | old.unresolved)
    newly_strict = new.strict - old.strict
    passed = not lost_separation and not newly_strict
    return passed, dict(passed=passed,
        lost_certified_separation_pairs=len(lost_separation),
        newly_strict_pairs=len(newly_strict),
        rejected_examples=[list(k) for k in sorted(lost_separation | newly_strict)[:10]],
        previous_strict_pairs=len(old.strict), new_strict_pairs=len(new.strict),
        previous_unresolved_pairs=len(old.unresolved), new_unresolved_pairs=len(new.unresolved),
        old_intersections_not_accepted=True, penetration_depth_comparison=False,
        swept_or_dynamics_checked=False, accepted_reference=False)


class WholeBodyStepGuard:
    def __init__(self, *, motion, motion_sha256, terrain):
        # The saved motion pins inventory reconstruction; scan() uses independent
        # FK of the proposed q/roots, never the saved motion's shape poses.
        self.K, pins, loaded, self.tree = load_case(motion, motion_sha256)
        self.names = [j.get('name') for j in ET.parse(TARGET).getroot().findall('joint') if j.get('type') != 'fixed']
        entries = []
        for link in ET.parse(TARGET).getroot().findall('link'):
            for i, c in enumerate(link.findall('collision')):
                entries.append((link.get('name')+'#'+str(i), self.K.origin_transform(c.find('origin'))))
        if [s.identity for _,s,*_ in loaded] != [name for name,_ in entries]:
            raise ValueError('Shape-order identity mismatch')
        self.inventory = [(link, solid, entry[1]) for (link,solid,*_),entry in zip(loaded,entries)]
        self.pairs = list(combinations(range(43), 2))
        terrain = Path(terrain).resolve()
        if sha(terrain) != '3aee33efea6793974d901dd0f7e46859e2d12d2dfd79f38f12c777686566c2de':
            raise ValueError('Unreviewed terrain bytes')
        crop = HERE.parent/'nonlinear_guard_20261118/box_triangle_guard.py'
        prior = json.loads((crop.parent/'FRESH_SAT_AUDIT.json').read_text())
        if sha(crop) != prior['code_sha256'][crop.name]:
            raise ValueError('Crop implementation changed')
        spec = importlib.util.spec_from_file_location('step_guard_pinned_crop', crop)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.triangles, self.raw_triangle_ids = module.cropped_triangles(self.K.read_triangles(terrain), [9,12])
        if self.triangles.shape != (8068,3,3):
            raise ValueError('Finite triangle inventory mismatch')
        self.low = self.triangles.min(axis=1); self.high = self.triangles.max(axis=1)
        pins[terrain] = sha(terrain); pins[crop] = sha(crop)
        self.pins = dict(pins)
        manifest = {str(p):v for p,v in pins.items()}
        for name in ['wholebody_step_guard.py','nominal_certificates.py','audit_self_pairs.py']:
            manifest[str(HERE/name)] = sha(HERE/name)
        self.binding = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()

    def verify_inputs(self):
        for path, expected in self.pins.items():
            if sha(path) != expected:
                raise ValueError('Guard input changed: '+str(path))

    def scan(self, model, q, roots, frame_indices):
        q = finite(q); roots = rigid(roots); frames = np.asarray(frame_indices)
        if list(model.names) != self.names:
            raise ValueError('Named29 joint binding mismatch')
        if q.shape != (len(roots),29) or frames.shape != (len(q),) or frames.dtype.kind not in 'iu':
            raise ValueError('Bounded named native-frame arrays required')
        if len(set(frames.tolist())) != len(frames) or np.any(frames<0) or np.any(frames>=997) or np.any(np.diff(frames)<=0):
            raise ValueError('Unique ordered native frame indices required')
        self.verify_inputs()
        world = {link:roots @ local for link,local in self.tree.fk(q,self.names).items()}
        posed = []
        for link, solid, local in self.inventory:
            poses = world[link] @ local
            lo, hi = solid.interval(np.broadcast_to(np.eye(3),(len(q),3,3)),poses)
            posed.append((link,solid,poses,lo,hi))
        strict=set(); unresolved=set(); records=0
        for a,b in self.pairs:
            la,sa,pa,alo,ahi=posed[a];lb,sb,pb,blo,bhi=posed[b]
            records += len(q)
            if la == lb:
                # Rigid compound parts are accounted, not independent bodies.
                continue
            ids=np.flatnonzero(np.all(ahi>=blo-1e-8,axis=1)&np.all(bhi>=alo-1e-8,axis=1))
            if not len(ids):continue
            result=classify(sa,pa[ids],sb,pb[ids])
            self._record(result,ids,frames,'self',a,b,strict,unresolved)
        for a,(_,solid,poses,lo,hi) in enumerate(posed):
            jobs=[]
            for start in range(0,len(q),64):
                stop=min(len(q),start+64)
                overlap=np.all(self.high[None]>=lo[start:stop,None]-1e-8,axis=2)&np.all(self.low[None]<=hi[start:stop,None]+1e-8,axis=2)
                fs,ts=np.nonzero(overlap);jobs.extend(zip((fs+start).tolist(),ts.tolist()))
            for start in range(0,len(jobs),512):
                batch=jobs[start:start+512];fs=np.array([f for f,t in batch]);ts=np.array([t for f,t in batch])
                result=classify_triangles(solid,poses[fs],self.triangles[ts])
                self._record(result,fs,frames,'terrain',a,ts,strict,unresolved)
        if records != 903*len(q):
            raise ArithmeticError('Incomplete self pair accounting')
        self.verify_inputs()
        return EndpointState(self.binding,tuple(map(int,frames)),frozenset(strict),frozenset(unresolved),records,43*len(q)*len(self.triangles))

    @staticmethod
    def _record(result, ids, frames, kind, first, second, strict, unresolved):
        masks=[np.asarray(result[k],bool) for k in ['separated','strict','unresolved']]
        if any(x.shape!=(len(ids),) for x in masks) or not np.all(sum(x.astype(int) for x in masks)==1):
            raise ArithmeticError('Incomplete or overlapping narrowphase classes')
        for mask,target in [(masks[1],strict),(masks[2],unresolved)]:
            for k in np.flatnonzero(mask):
                b=int(second[k]) if isinstance(second,np.ndarray) else int(second)
                target.add((kind,int(frames[ids[k]]),int(first),b))

    compare = staticmethod(compare_states)