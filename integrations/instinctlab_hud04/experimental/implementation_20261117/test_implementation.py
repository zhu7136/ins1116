"""CPU/unit tests only. Simulator API stubs do not establish GPU acceptance."""
import copy,hashlib,importlib.util,json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).parent
OVERLAY=ROOT/'base/overlay/instinctlab'
if not OVERLAY.is_dir():OVERLAY=ROOT.parents[1]/'overlay/instinctlab'

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

M=module('cpu_hud04_motion',OVERLAY/'assets/hud04_motion.py')
from joint_temporal_fit import fit,residuals

class ImplementationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT)
        self.path=Path(self.temp.name)/'synthetic-retargeted.npz'
        self.names=['body_joint_'+str(i) for i in range(29)]
        self.limits={n:dict(lower=-2,upper=2,velocity=10) for n in self.names}
        self.q=np.tile(np.arange(29)*.001,(997,1))
        self.p=np.zeros((997,3));self.r=np.tile([1.,0,0,0],(997,1))
    def tearDown(self):self.temp.cleanup()
    def write(self,**kw):
        data=dict(joint_pos=self.q,joint_names=np.array(self.names),base_pos_w=self.p,base_quat_w=self.r,framerate=90)
        data.update(kw);np.savez_compressed(self.path,**data);return self.path
    def read(self,names=None):return M.read_named_motion(self.path,self.names if names is None else names,self.limits)
    def test_named_order(self):
        self.write();q,p,r,fps=self.read(self.names[::-1]);np.testing.assert_array_equal(q,self.q[:,::-1]);self.assertEqual(fps,90)
    def test_duplicate_names(self):
        self.write(joint_names=np.array([self.names[0]]*29))
        with self.assertRaises(ValueError):self.read()
    def test_head_rejected(self):
        names=self.names.copy();names[0]='head_yaw_joint'
        with self.assertRaises(ValueError):M.permutation(names,names)
    def test_pickle_rejected(self):
        self.write(joint_names=np.array(self.names,dtype=object))
        with self.assertRaises(ValueError):self.read()
    def test_no_resampling(self):
        self.write(framerate=50)
        with self.assertRaises(ValueError):self.read()
    def test_no_frame_trimming(self):
        self.write(joint_pos=self.q[:996])
        with self.assertRaises(ValueError):self.read()
    def test_no_angle_clip(self):
        self.q[12,0]=3;self.write()
        with self.assertRaises(ValueError):self.read()
    def test_no_speed_clip(self):
        self.q[12,0]=1;self.write()
        with self.assertRaises(ValueError):self.read()
    def test_nonfinite(self):
        self.p[0,0]=np.nan;self.write()
        with self.assertRaises(ValueError):self.read()
    def test_no_quaternion_normalization(self):
        self.r[0,0]=2;self.write()
        with self.assertRaises(ValueError):self.read()
    def test_joint_fit_not_reference(self):
        self.path=Path(self.temp.name)/'step1.joint_fit.npz';self.write()
        with self.assertRaises(ValueError):self.read()
    def test_lp_couples_time_not_pointwise_clip(self):
        source=np.array([[0.],[1.],[0.]])
        q,report=fit(source,[-2.],[2.],[9.])
        np.testing.assert_allclose(q[:,0],[0,.1,0],atol=1e-8)
        self.assertEqual(report['solver_calls'],1);self.assertFalse(report['posthoc_angle_clipping'])
        self.assertFalse(report['accepted_reference']);self.assertLess(max(report['residuals'].values()),1e-6)
    def test_lp_preserves_feasible_input(self):
        q=np.array([[.1],[.11],[.12]])
        out,_=fit(q,[-1],[1],[5]);np.testing.assert_allclose(out,q,atol=1e-8)
    def test_support_boundaries_fail_closed(self):
        body=['left_ankle_roll_link','left_knee_link','head_link']
        frames=np.array([-1,0,5,6,996,997])
        masks=M.allowed_support_mask(body,frames,{'left_ankle_roll_link':[[0,5]],'left_knee_link':[[996,996]]})
        np.testing.assert_array_equal(masks[:,0],[0,1,1,0,0,0]);np.testing.assert_array_equal(masks[:,1],[0,0,0,0,1,0]);self.assertFalse(masks[:,2].any())
    def support(self):
        return dict(schema_version=1,status='reviewed_support_windows',frames=997,framerate=90,
            motions={'synthetic-retargeted.npz':{n:[] for n in M.SUPPORT_LINKS}})
    def test_support_requires_all_six_links(self):
        x=self.support();del x['motions']['synthetic-retargeted.npz']['left_knee_link']
        with self.assertRaises(ValueError):M.validate_support(x,['synthetic-retargeted.npz'])
    def test_support_rejects_overlap(self):
        x=self.support();x['motions']['synthetic-retargeted.npz'][M.SUPPORT_LINKS[0]]=[[0,5],[5,7]]
        with self.assertRaises(ValueError):M.validate_support(x,['synthetic-retargeted.npz'])
    def test_support_rejects_unreviewed(self):
        x=self.support();x['status']='candidate'
        with self.assertRaises(ValueError):M.validate_support(x,['synthetic-retargeted.npz'])
    def test_support_accepts_explicit_empty_schedules(self):
        x=self.support();self.assertEqual(M.validate_support(x,['synthetic-retargeted.npz']),x)
    def test_terrain_bundle_binds_checksums_and_rejects_escape(self):
        root=Path(self.temp.name)
        def entry(name,raw):
            (root/name).write_bytes(raw)
            return dict(path=name,sha256=hashlib.sha256(raw).hexdigest())
        # Synthetic schema fixture only, not a fabricated accepted real motion.
        support=entry('support.json',json.dumps(self.support()).encode())
        terrain=entry('synthetic.stl',b'SYNTHETIC UNIT TEST NOT A MESH')
        metadata=entry('metadata.yaml',b'motion_files:\n- motion_file: synthetic-retargeted.npz\n  terrain_id: 0\nterrains:\n- terrain_file: synthetic.stl\n  terrain_id: 0\n')
        manifest=dict(status='accepted_for_terrain_replay',review_evidence={k:'SYNTHETIC TEST ONLY' for k in ('terrain','source_conventions','support_intervals')},
            files=[dict(path='synthetic-retargeted.npz')],terrain_metadata=metadata,terrain_mesh=terrain,support_schedule=support)
        path=root/'manifest.json';path.write_text(json.dumps(manifest));M.validate_terrain_bundle(path)
        manifest['terrain_mesh']['path']='../outside.stl';path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):M.validate_terrain_bundle(path)
    def test_all_overlay_sources_compile(self):
        files=list(OVERLAY.rglob('*.py'));self.assertEqual(len(files),10)
        for p in files:compile(p.read_bytes(),str(p),'exec')
    def runtime_module(self):
        # Real adapter functions, torch tensors and NPZ reader, but synthetic
        # Isaac/Instinct base classes. NOT a live simulator instantiation.
        modules={}
        for name in ['isaaclab','isaaclab.utils','instinctlab','instinctlab.assets',
            'instinctlab.assets.hud04','instinctlab.assets.hud04_contract','instinctlab.motion_reference',
            'instinctlab.motion_reference.motion_files']:
            modules[name]=types.ModuleType(name);modules[name].__path__=[]
        modules['isaaclab.utils'].configclass=lambda cls:cls
        modules['instinctlab.assets.hud04'].HUD04_URDF='synthetic'
        modules['instinctlab.assets.hud04_contract'].validate_model=lambda p:(self.names,self.limits)
        modules['instinctlab.assets.hud04_motion']=M
        for leaf,cls in [('amass_motion','AmassMotion'),('amass_motion_cfg','AmassMotionCfg'),('terrain_motion','TerrainMotion'),('terrain_motion_cfg','TerrainMotionCfg')]:
            k='instinctlab.motion_reference.motion_files.'+leaf;modules[k]=types.ModuleType(k);setattr(modules[k],cls,type(cls,(),{}))
        with patch.dict(sys.modules,modules):
            return module('instinctlab.motion_reference.motion_files.hud04_motion',OVERLAY/'motion_reference/motion_files/hud04_motion.py')
    def test_actual_adapter_dispatch_reorders(self):
        import torch
        r=self.runtime_module();obj=r.HUD04AmassMotion();self.write()
        obj.cfg=types.SimpleNamespace(motion_interpolate_func=None,skip_frames=0,clip_joint_ref_to_robot_limits=False,
            ensure_link_below_zero_ground=False,motion_start_height_offset=0)
        obj.buffer_device='cpu';obj.isaac_joint_names=self.names[::-1];obj._all_motion_files=[str(self.path)]
        obj._pack_retargetted_motion_sequence=lambda *args:args
        p,quat,q,fps=obj._read_motion_file(0)
        np.testing.assert_allclose(q.numpy(),self.q[:,::-1],atol=1e-8);self.assertEqual(fps,90)
        obj.cfg.skip_frames=1
        with self.assertRaises(ValueError):obj._read_motion_file(0)
    def test_actual_adapter_source_clock_uses_reset_stub(self):
        import torch
        r=self.runtime_module();obj=r.HUD04TerrainMotion();obj.buffer_device='cpu';obj.output_device='cpu'
        obj.env_ids_to_assigned_ids=lambda ids:ids
        obj._assigned_env_motion_selection=torch.tensor([0,1]);obj._motion_buffer_start_time_s=torch.tensor([.2,.5])
        obj._all_motion_sequences=types.SimpleNamespace(framerate=torch.tensor([90.,90.]))
        frames=obj.support_source_frames(torch.tensor([0,1]),torch.tensor([.02,.04]))
        np.testing.assert_array_equal(frames.numpy(),[20,49])
    def test_existing_real_fit_independent_residual(self):
        report=json.loads((ROOT/'STEP1_JOINT_FIT_RESULT.json').read_text())
        model_path=Path(os.environ.get('HUD04_DELIVERY_ROOT','/workspace/hud04-recovery-bf4a057'))/'flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf'
        import xml.etree.ElementTree as ET
        xml={j.get('name'):j.find('limit').attrib for j in ET.parse(model_path).getroot().findall('joint') if j.get('type')=='revolute'}
        with np.load(ROOT/'step1.joint_fit.npz',allow_pickle=False) as z:q=z['joint_pos'];names=z['joint_names'].tolist()
        lo,hi,v=[np.array([float(xml[n][k]) for n in names]) for k in ('lower','upper','velocity')]
        self.assertEqual(q.shape,(997,29));self.assertLess(max(residuals(q,lo,hi,v).values()),1e-6)
        self.assertEqual(report['source_position_bad_frames'],261);self.assertEqual(report['source_overspeed_intervals'],165)
    def test_real_runtime_order_regression(self):
        delivery=Path(os.environ.get('HUD04_DELIVERY_ROOT','/workspace/hud04-recovery-bf4a057'))
        path=delivery/'integrations/instinctlab_hud04/experimental/import_observation_20261117/RECOVERED_RUNTIME_INVENTORY_REVIEW_20261117.json'
        if not path.is_file():path=Path('/workspace/hud04-runtime-progress-20261117/RECOVERED_RUNTIME_INVENTORY_REVIEW_20261117.json')
        review=json.loads(path.read_text())
        stored=review['model_joint_order'];runtime=review['runtime_joint_order']
        self.assertNotEqual(stored,runtime)
        q=np.arange(29);mapped=q[M.permutation(stored,runtime)]
        np.testing.assert_array_equal(mapped[M.permutation(runtime,stored)],q)
    def test_actual_conditional_contact_adapter(self):
        import torch
        asset=types.ModuleType('instinctlab.assets.hud04_motion');asset.allowed_support_mask=M.allowed_support_mask
        with patch.dict(sys.modules,{'instinctlab.assets.hud04_motion':asset}):
            c=module('tested_conditional_contacts',OVERLAY/'tasks/shadowing/perceptive/config/hud04/conditional_contacts.py')
        buffer=types.SimpleNamespace(env_ids_is_assigned=lambda ids:torch.tensor([True,True]),
            support_source_frames=lambda ids,t:torch.tensor([20,49]),get_current_motion_identifiers=lambda ids:['/synthetic/m.npz']*2,
            cfg=types.SimpleNamespace(path='/synthetic'))
        manager=types.SimpleNamespace(_motion_buffers={'TerrainMotion':buffer},_timestamp=torch.tensor([.02,.04]))
        sensor=types.SimpleNamespace(body_names=['left_knee_link','head_link'],
            data=types.SimpleNamespace(net_forces_w=torch.tensor([[[0.,0.,1000.],[0.,0.,0.]],[[0.,0.,1000.],[0.,0.,0.]]])))
        class Scene(dict):pass
        scene=Scene(motion_reference=manager);scene.sensors={'contact_forces':sensor}
        env=types.SimpleNamespace(scene=scene,num_envs=2,device='cpu',cfg=types.SimpleNamespace(
            hud04_support_schedule={'motions':{'m.npz':{'left_knee_link':[[20,20]]}}}))
        sensor_cfg=types.SimpleNamespace(name='contact_forces')
        result=c.illegal_support_contact(env,500,sensor_cfg)
        np.testing.assert_array_equal(result.numpy(),[False,True])
        sensor.data.net_forces_w[0,1,2]=1000.
        self.assertTrue(c.illegal_support_contact(env,500,sensor_cfg)[0].item())

if __name__=='__main__':unittest.main(verbosity=2)