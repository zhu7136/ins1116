# HUD04 浮动基座、足部接触与关节动作映射：待 GPU 验证

独立、无参考动作的资产级诊断，不是完整 InstinctLab 环境或训练入口。
新增目录，不修改已验证的 `../hud04_asset_camera_probe/`。
**本版只有 CPU 软件测试结果，没有执行新的 GPU 任务。**

## 恢复的集成约束

- 29 个身体自由度；头 yaw/pitch 继续在 URDF 模型零位结构性固定。
- 不把 G1 的 29 维策略权重或列顺序当成 HUD04 的控制语义。
- 已有完整参考动作尚未验收；此探针不加载、导出、修改或应用任何动作候选。
- 不执行 LP/QP、优化器 setup 或训练。此前求解预算不由“继续”自动续期。
- 完整 Plane/Terrain 感知任务仍需真实兼容且验收通过的参考动作；地形任务还需配套 metadata 和网格。
- GPU 创建、启动、停止及计费必须在平台层单独核对。脚本内部看门狗不是总账单上限。

## 明确的动作—观测契约

与 InstinctLab `de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057` 和 IsaacLab `v2.3.2` 静态源码对照：

1. `JointPositionActionCfg(preserve_order=False)` 令策略动作使用**运行时 articulation 顺序**；不是 URDF 文件顺序。
2. 每个关节的目标为 `default_joint_pos + raw_action * named_scale`。比例系数按关节名称解析。
   本探针不增加静默限幅，配置 `clip=None`；诊断输入幅度另行限制。
3. `joint_pos_rel` / `joint_vel_rel` 使用相同运行时顺序；默认关节速度为零。
   `last_action` 是原始动作，不能错误替换成缩放后的关节目标。
4. InstinctLab 的 `AmassMotion._read_retargetted_motion_file` 已将数据列按 `isaac_joint_names` 重排，
   `JointPosRefCommand` 再减默认姿态。未来完整集成必须继续断言参考管理器和机器人名称顺序一致。
5. 这里只验证一帧不加噪的三个关节观测项，不声称包含历史、参考命令、critic 特征或深度图。

现有 GPU 报告的 PhysX 顺序与 URDF 顺序确实不同。CPU 测试采用该实际顺序作为回归夹具，
另测随机置换的双向映射，拒绝重复名、缺失名和头部列。
旧适配器默认的运行时顺序不因此被判定有 bug；这里使隐式约定可检查、可记录。

## 新探针计划

- 一个真实 IsaacLab `ManagerBasedEnv`，不创建 RL 环境，无策略、优化器或奖励。
- 使用已验证资产目录中的相同 URDF、碰撞体、显式 DelayedPD、8/4 求解器迭代设置。
  `fix_base=False`，开启重力和接触；本地静态地面，无外部 Nucleus 资源。
- 真实 ActionManager 的 29 个坐标分别做正负 0.05 原始动作的处理检查，共 58 次。
  这些检查不推进物理，不等于 29 个驱动器的方向响应均已验证。
- 两次确定性 reset，各最多 50 个控制步，decimation=4，物理 dt=0.005 秒；
  合计最多 400 物理步 / 2 秒仿真时间。主要保持启动姿态，第 21–30 控制步给左腕一个很小的命令。
- 用原 URDF 足盒和解析 FK 计算启动高度：根坐标系最低点为 -0.895369085 米，
  根高度取 0.905369085 米，使最低脚底初始在地面上方 1 厘米。
  这是几何初始化，不是 IK、平衡优化或新参考动作；并不沿用未经接触验证的 0.9 米硬编码高度。
- 逐控制步比较真实处理后动作、三个真实观测项、头部零位及有限值。
  记录足部净力、非足部净力、根姿态、关节误差、力矩比例、关节速度和硬限位越界。
- 根高度低于 0.35 米或倾角超过约 60° 时提前结束该 episode 并记录失败，不强行维持或重启。

`probe_passed` 仅表示两个短 episode 未触发跌倒保护、均观察到双足向上净力，
且硬关节限位越界小于诊断容差 0.05 rad。这个容差不是硬件安全标准。
力矩、跟踪误差和非足部接触保留为待审指标，**此标志不能代替它们的物理验收**。
身体净接触力可能包含自接触，不是经过地面过滤的足底力；不能据此声称站立稳定、无穿透或动力学正确。
无论结果如何，`training_ready=false`。不加载相机，因此也不重复固定基座深度/退出诊断。

## 默认只打印计划

在本目录运行（不会初始化 Isaac/CUDA）：

```sh
python -m unittest -v test_cpu.py
python probe.py
python worker.py
```

本版 CPU 检查：新增 23 项全部通过；原资产包 34 项回归通过。
这些是软件测试，不是物理、接触、站姿或真实管理器运行的证据。

在已核对任务状态、源码提交、资源和时间边界后才可显式启动：

```sh
python worker.py --execute-probe --expected-commit "$(git rev-parse HEAD)"
# 若镜像需要专用 Python 启动器，显式添加 --isaac-launcher /已核实的路径
```

Flux 候选入口为 `gm-run ins1116/flux/hud04_floating_control_probe/worker.py --execute-probe --expected-commit 固定的40位提交哈希`，
必须先回读当前平台代码配置；不能把这条候选命令当成已经提交的任务。
平台配置必须传入本次已核实提交的**字面值**，不能在远端用 `git rev-parse` 动态代替预期提交。
worker 在初始化 GPU 前拒绝缺失或不匹配的提交。
只允许一个子进程，无自动重试；初始化 300 秒、子进程 540 秒、关闭 30 秒上限。
不在线安装或升级环境。已有结果/日志时拒绝覆盖。

新目录 `manifest.json` 校验可执行输入；原目录仍使用原 manifest 独立校验，且保持不变。
子进程报告（含失败的接触/控制验收）通过一个最多 32 KiB 的带 SHA256 日志事件回传，
`worker.verify_evidence` 可离线校验。只有正常退出、观察到上下文清理完成标记且诊断通过才令 worker 成功。
截止后终止只是止损，不作为正常退出通过；CPU 校验未证明新的日志事件已完成平台端到端传输。

## 源码核对与许可

- InstinctLab 基线：`de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057`，
  `tasks/shadowing/perceptive/perceptive_env_cfg.py`、`motion_files/amass_motion.py`、
  `envs/mdp/commands/shadowing_command.py`。沿用其 CC BY-NC 4.0 非商业与署名条件。
- IsaacLab `v2.3.2`：`envs/manager_based_env.py`、`manager_based_env_cfg.py`、
  `envs/mdp/actions/{actions_cfg,joint_actions}.py`、`assets/articulation/articulation.py`。
  `ManagerBasedEnv.close()` 的公开清理流程包含 `sim.clear_instance()`；随后正常关闭 app。
  静态对照不是新探针在 Flux 镜像上成功的证明。
- HUD04 资产及初始控制参数沿用原包，来源、Apache-2.0 模型许可和限制见原包 README / licenses。

此目录自身没有提交 Flux 任务、执行求解器或 PPO 的自动入口；只有显式参数能启动 GPU 诊断。