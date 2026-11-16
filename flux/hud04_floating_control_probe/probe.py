"""Bounded floating-base/control/contact diagnostic, NOT a training task.

Uses real IsaacLab ManagerBasedEnv Action/Observation managers. No camera,
reference data, optimizer, reward, policy or InstinctLab task registration.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

from control_contract import FEET, foot_box_bottoms, permutation, position_targets, proprioception, read_model

ROOT = Path(__file__).resolve().parent
ASSET = ROOT.parent / 'hud04_asset_camera_probe'


def event(stage, **kw):
    print(json.dumps(dict(stage=stage, **kw), allow_nan=False), flush=True)


def main(execute=False, output=None):
    if not execute:
        event('plan_only', gpu_started=False, physics_steps_max=400, physics_dt_s=.005,
              control_decimation=4, episodes=2, camera_enabled=False, training_updates=0,
              lp_solves=0, qp_solves=0, optimizer_setup_calls=0)
        return
    sys.path.insert(0, str(ASSET))
    from preflight import audit
    from probe_checks import check_head_zero
    audit(ASSET)  # preserves the entire verified fixed-base package unchanged
    output = Path(output or ROOT / 'output')
    output.mkdir(exist_ok=False)
    os.environ['HUMANOID_DESCRIPTION_DIR'] = str(ASSET / 'humanoid-description')
    os.environ['ENABLE_CAMERAS'] = '0'
    event('isaac_initialization_started')
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False, device='cuda:0').app
    env = None
    try:
        import numpy as np
        import torch
        import isaaclab.sim as sim_utils
        import isaaclab.envs.mdp as mdp
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
        from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.utils import configclass
        from isaaclab.utils.math import subtract_frame_transforms

        asset = runpy.run_path(str(ASSET / 'asset_config.py'))
        urdf_names, model = read_model(asset['HUD04_URDF'])
        bottoms = foot_box_bottoms(asset['HUD04_URDF'])
        root_height = .01 - min(bottoms.values())
        robot_cfg = asset['HUD04_CFG'].replace(prim_path='{ENV_REGEX_NS}/Robot')
        robot_cfg.spawn.fix_base = False
        robot_cfg.init_state.pos = (0., 0., root_height)

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = robot_cfg
            # Local static cuboid: avoids external ground-plane assets/materials.
            ground = AssetBaseCfg(
                prim_path='/World/ground',
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., -.05)),
                spawn=sim_utils.CuboidCfg(size=(6., 6., .1),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1., dynamic_friction=1., restitution=0.)))
            contacts = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',
                                        update_period=0., history_length=1, debug_vis=False)

        @configclass
        class ActionsCfg:
            joint_pos = mdp.JointPositionActionCfg(
                asset_name='robot', joint_names=urdf_names, preserve_order=False,
                scale=asset['HUD04_ACTION_SCALE'], use_default_offset=True, clip=None)

        @configclass
        class JointObsCfg(ObservationGroupCfg):
            joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
            joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
            last_action = ObservationTermCfg(func=mdp.last_action)
            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = False

        @configclass
        class ObservationsCfg:
            policy = JointObsCfg()

        cfg = ManagerBasedEnvCfg(scene=SceneCfg(num_envs=1, env_spacing=4., replicate_physics=False),
            actions=ActionsCfg(), observations=ObservationsCfg(), decimation=4, seed=11,
            sim=sim_utils.SimulationCfg(dt=.005, device='cuda:0', render_interval=4),
            ui_window_class_type=None, wait_for_textures=False)
        env = ManagerBasedEnv(cfg)
        env.reset()
        robot, contacts = env.scene['robot'], env.scene['contacts']
        names = list(robot.joint_names)
        permutation(urdf_names, names)
        if robot.is_fixed_base:
            raise AssertionError('Expected floating articulation')
        term = env.action_manager.get_term('joint_pos')
        if term.action_dim != 29 or list(term.IO_descriptor.joint_names) != names:
            raise AssertionError('Actual action manager order differs from articulation')
        expected_defaults = np.array([model[n]['default'] for n in names])
        defaults = robot.data.default_joint_pos[0].cpu().numpy().copy()
        np.testing.assert_allclose(defaults, expected_defaults, atol=1e-6)
        scales = np.array([model[n]['scale'] for n in names])
        np.testing.assert_allclose(scales, [asset['HUD04_ACTION_SCALE'][n] for n in names])
        effort = np.array([model[n]['effort'] for n in names])
        lower = np.array([model[n]['lower'] for n in names])
        upper = np.array([model[n]['upper'] for n in names])
        mapping_error = 0.
        # Exercise real affine processing at every coordinate without stepping physics.
        # This validates routing, NOT physical directional response for all joints.
        for index in range(29):
            for sign in (-1., 1.):
                a = np.zeros(29); a[index] = sign * .05
                expected = position_targets(a, names, model, defaults)
                if np.any(expected < lower) or np.any(expected > upper):
                    raise AssertionError('Diagnostic command exceeds URDF limits')
                env.action_manager.process_action(torch.tensor(a[None], device=env.device, dtype=torch.float32))
                actual = term.processed_actions[0].cpu().numpy()
                mapping_error = max(mapping_error, float(np.max(np.abs(actual - expected))))
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
        parent_ids = [robot.body_names.index(n) for n in ['waist_pitch_link', 'head_yaw_link']]
        child_ids = [robot.body_names.index(n) for n in ['head_yaw_link', 'head_pitch_link']]
        foot_ids = [contacts.body_names.index(n) for n in FEET]
        nonfoot_ids = [i for i, n in enumerate(contacts.body_names) if n not in FEET]
        event('asset_initialized', body_dof=29, fixed_base_diagnostic=False,
              initial_root_height_m=root_height, manager_affine_checks=58)
        records, episodes, total_steps = [], [], 0
        for episode in range(2):
            env.reset()
            np.testing.assert_allclose(robot.data.joint_pos[0].cpu().numpy(), defaults, atol=1e-5)
            np.testing.assert_allclose(robot.data.root_pos_w[0].cpu().numpy(), [0, 0, root_height], atol=1e-5)
            if float(env.action_manager.action.abs().max()) > 1e-7:
                raise AssertionError('Raw action not reset')
            seen = np.zeros(2, dtype=bool)
            fall, peak_tracking, peak_effort, peak_nonfoot, peak_speed = False, 0., 0., 0., 0.
            max_limit_violation = 0.
            for step in range(50):
                if not app.is_running():
                    raise RuntimeError('Simulator stopped early')
                a = np.zeros(29)
                if 20 <= step < 30:
                    a[names.index('left_wrist_yaw_joint')] = .05
                action = torch.tensor(a[None], device=env.device, dtype=torch.float32)
                obs, _ = env.step(action)
                total_steps += 4
                q = robot.data.joint_pos[0].cpu().numpy()
                qd = robot.data.joint_vel[0].cpu().numpy()
                expected = position_targets(a, names, model, defaults)
                np.testing.assert_allclose(term.processed_actions[0].cpu().numpy(), expected, atol=1e-6)
                reference_obs = proprioception(q, qd, a, names, model, defaults)
                for key, value in reference_obs.items():
                    np.testing.assert_allclose(obs['policy'][key][0].cpu().numpy(), value, atol=1e-6, rtol=1e-6)
                head_pos, head_quat = subtract_frame_transforms(
                    robot.data.body_pos_w[:, parent_ids], robot.data.body_quat_w[:, parent_ids],
                    robot.data.body_pos_w[:, child_ids], robot.data.body_quat_w[:, child_ids])
                check_head_zero(head_pos[0].cpu().tolist(), head_quat[0].cpu().tolist())
                forces = contacts.data.net_forces_w[0].cpu().numpy()
                root = robot.data.root_pos_w[0].cpu().numpy()
                quat = robot.data.root_quat_w[0].cpu().numpy()
                torque = robot.data.applied_torque[0].cpu().numpy()
                if not all(np.isfinite(v).all() for v in [q, qd, forces, root, quat, torque]):
                    raise AssertionError('Nonfinite dynamics')
                fz = forces[foot_ids, 2]
                seen |= fz > 1.
                peak_nonfoot = max(peak_nonfoot, float(np.linalg.norm(forces[nonfoot_ids], axis=1).max(initial=0)))
                peak_tracking = max(peak_tracking, float(np.max(np.abs(q - expected))))
                peak_effort = max(peak_effort, float(np.max(np.abs(torque) / effort)))
                peak_speed = max(peak_speed, float(np.max(np.abs(qd))))
                max_limit_violation = max(max_limit_violation, float(np.maximum(lower-q, q-upper).max(initial=0)))
                up_z = 1 - 2 * (quat[1]**2 + quat[2]**2)
                fall = bool(root[2] < .35 or up_z < .5)
                if step % 5 == 0 or fall:
                    records.append(dict(episode=episode, control_step=step+1,
                        root_xyz_m=root.tolist(), root_up_z=float(up_z), foot_net_fz_N=fz.tolist(),
                        tracking_error_max_rad=float(np.max(np.abs(q-expected)))))
                if fall:
                    break
            episodes.append(dict(episode=episode, control_steps=step+1, fall_guard_triggered=fall,
                foot_vertical_contact_seen=seen.tolist(), max_tracking_error_rad=peak_tracking,
                max_effort_limit_ratio=peak_effort, max_joint_speed_rad_s=peak_speed,
                max_nonfoot_net_contact_N=peak_nonfoot, max_hard_joint_limit_violation_rad=max_limit_violation))
        accepted = all(not e['fall_guard_triggered'] and all(e['foot_vertical_contact_seen'])
                       and e['max_hard_joint_limit_violation_rad'] < .05 for e in episodes)
        report = dict(stage='floating_control_diagnostic_completed', diagnostic_completed=True,
            probe_passed=accepted, contact_control_acceptance_passed=accepted,
            training_ready=False, training_updates=0, motion_files_used=0, lp_solves=0, qp_solves=0,
            optimizer_setup_calls=0, body_dof=29, head_policy_dof=0, fixed_base_diagnostic=False,
            action_order='runtime_articulation_order', joint_names=names,
            urdf_to_runtime_gather=permutation(urdf_names, names),
            manager_affine_checks=58, manager_mapping_max_error_rad=mapping_error,
            reset_checks=2, physics_steps=total_steps, physics_dt_s=.005, control_decimation=4,
            initial_root_height_m=root_height, initial_foot_box_bottoms_root_m=bottoms,
            episodes=episodes, trace=records,
            limitations=['No reference motion, learning, or InstinctLab task execution',
                'Unnoised joint observations only; no depth or history pipeline',
                '58 action checks validate manager routing, not physical response of every actuator',
                'Net body contact forces include self contact; not a ground-only force measurement',
                'One-second episodes and fixed provisional PD do not establish robust standing'])
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        event(**report)
        if not accepted:
            raise RuntimeError('Diagnostic recorded; contact/control acceptance not passed')
    finally:
        event('simulator_shutdown_started', wait_for_replicator=False)
        try:
            if env is not None:
                env.close()  # documented public cleanup includes sim.clear_instance()
            else:
                from isaaclab.sim import SimulationContext
                if SimulationContext.instance() is not None:
                    SimulationContext.instance().clear_instance()
            event('simulation_context_cleanup_completed')
        finally:
            app.close(wait_for_replicator=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-probe', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    main(args.execute_probe, args.output)