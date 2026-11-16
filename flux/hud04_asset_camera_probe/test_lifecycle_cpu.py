"""CPU-only validation of diagnostic controls, not Isaac runtime proof."""
import json,subprocess,sys,unittest
from pathlib import Path
from lifecycle_probe import deadline,phases,MODES,normal_process_exit

class LifecycleTests(unittest.TestCase):
    def test_close_can_exit_without_returning(self):
        stages=['lifecycle_app_initialized','lifecycle_app_closing']
        self.assertTrue(normal_process_exit(None,0,stages,'app-only',None))
        self.assertFalse(normal_process_exit(None,0,stages,'empty-physics',None))

    def test_watchdog_zero_exit_still_fails(self):
        stages=['lifecycle_app_initialized','lifecycle_app_closing']
        self.assertFalse(normal_process_exit('normal_shutdown_deadline',0,stages,'app-only',None))
        self.assertFalse(normal_process_exit(None,1,stages,'app-only',None))

    def test_public_context_cleanup_precedes_app_close(self):
        for name in ['lifecycle_probe.py','probe.py']:
            text=Path(__file__).with_name(name).read_text()
            self.assertLess(text.index('sim.clear_instance()'),text.index('app.close(wait_for_replicator=False)'))

    def test_plan_no_simulator(self):
        p=subprocess.run([sys.executable,str(Path(__file__).with_name('lifecycle_probe.py'))],
                         capture_output=True,text=True,check=True,timeout=5)
        data=json.loads(p.stdout)
        self.assertFalse(data['gpu_started']);self.assertEqual(data['training_updates'],0)
        self.assertEqual(data['modes'],['app-only','empty-physics'])

    def test_phase_prefix_and_partial(self):
        text='SDK prefix '+json.dumps(dict(stage='lifecycle_app_closing',mode='app-only'))+'\n{"stage": "lifecycle_incomplete'
        self.assertEqual(phases(text),[dict(stage='lifecycle_app_closing',mode='app-only')])

    def test_shutdown_timeout_not_success(self):
        self.assertIsNone(deadline(31,True,29.9))
        self.assertEqual(deadline(32,True,30),'normal_shutdown_deadline')

    def test_initialization_and_total_limits(self):
        self.assertEqual(deadline(90,False,None),'app_initialization_deadline')
        self.assertIsNone(deadline(90,True,None))
        self.assertEqual(deadline(120,True,None),'child_wall_limit')

    def test_controls_exclude_robot_and_sensor(self):
        text=Path(__file__).with_name('lifecycle_probe.py').read_text()
        self.assertEqual(len(MODES),2)
        self.assertNotIn('from isaaclab.assets',text)
        self.assertNotIn('from isaaclab.sensors',text)
        self.assertIn('app.close(wait_for_replicator=False)',text)
        self.assertIn('start_new_session=True',text)

if __name__=='__main__':unittest.main()
