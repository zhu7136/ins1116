# IsaacLab 2.3.2 静态兼容性检查（非运行结果）

检查时间：2026-11-16。依据公开上游`isaac-sim/IsaacLab`的`v2.3.2`标签14份源码；来源文件的SHA256见SOURCE_MANIFEST.json。
这不证明Flux `isaac-lab2.3.2-v3`镜像与上游源码完全一致，仍需GPU验证。

| 本探针用法 | 已对照的上游实现 |
|---|---|
| `RayCasterCameraCfg`, `mesh_prim_paths`, offset | `sensors/ray_caster/ray_caster_camera_cfg.py` |
| `[N,H,W,1]`深度、`CameraData`、camera frame | `sensors/ray_caster/ray_caster_camera.py` |
| `PinholeCameraPatternCfg` 48×27 | `sensors/ray_caster/patterns/patterns_cfg.py` |
| 相机四元数`convention='world'`的轴向转换 | `utils/math.py`及`ray_caster_camera.py` |
| `DelayedPDActuatorCfg`、力矩/速度限制字段定义 | `actuators/actuator_pd_cfg.py`及`actuator_base_cfg.py`；探针资产沿用显式DelayedPD，不称作隐式PD |
| `UrdfFileCfg`及`UrdfConverterCfg.JointDriveCfg` | `sim/spawners/from_files/from_files_cfg.py`及`sim/converters/urdf_converter_cfg.py` |
| `fix_base=True`, `merge_fixed_joints=False`, `force_usd_conversion` | `sim/converters/urdf_converter_cfg.py` |
| `robot.data.default_joint_pos`、`write_joint_state_to_sim`、`.reset()` | `assets/articulation/articulation.py` |
| 写入关节状态后`sim.forward()` | `sim/simulation_context.py` |

发现并修订：创建默认姿态缓冲区不等同于将姿态写入PhysX。现在reset后先写入默认关节位置、速度，再forward/update。
增加头部导入零位检查、非有限值拒收及相机帧更新检查。

## 仍未验证

- GPU/PhysX能否成功导入该URDF和网格；关节树的实际排列及导入后刚体是否保留。
- Warp射线深度能否命中目标、深度输出是否满足有效率及形状要求。
- 镜像实际Python/IsaacSim版本、IsaacLab与源码是否一致；本地CPU不冒充Isaac运行环境。
- 真正自由基座的全身平衡、碰撞稳定性、域随机化、PPO训练及技能成功率。
- Flux完整日志获取、结果文件校验、任务停止后的资源释放和最终费用。

此探针固定根基座，只用于资产、头部锁定和胸口相机验证；通过也不表示机器人已经具备平衡或感知技能。