# InstinctLab 的 HUD04 专用集成（第一版候选）

这不是再跑一次独立资产探针。这里将已验证的 HUD04 资产接到原 InstinctLab
的感知运动跟踪环境，新增 HUD04 Plane / PLAY 配置，**不修改 G1**。

## 状态与边界

- 上游仓库：<https://github.com/project-instinct/InstinctLab>，固定提交
  `de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057`；候选环境为 IsaacLab v2.3.2。
- 已通过 36 项 CPU 软件测试；已在该提交的真实、稀疏 Git 检出中安装并逐文件回读，
  原有受跟踪源码无变化。稀疏检出仅用于安装检查，不是可运行的完整仿真环境。
- 已验证探针对应源 `8b421f55`，这里按 SHA256 复用其资产、控制契约和相机模块，
  不重写 PD、不改旧探针、不运行旧创建/发布/启动脚本。
- **尚未在 IsaacLab 中实际导入/构造此配置，也未执行完整环境 GPU 步进。**
  当前 CPU 环境没有 IsaacLab；合成测试不等于配置运行或物理接触验证。
- 没有获验收的 HUD04 参考动作、没有 PPO、没有技能学习结果；`training_ready=false`。
- 不创建 Flux 任务，不调用求解器，不在线安装依赖，没有自动训练/重试。
- 已完成的 `TASK_20261116_732` 和 `TASK_20261117_016` 不应重新提交。

## 文件与行为

`overlay/instinctlab/` 下六个文件将新增到原项目的
`source/instinctlab/instinctlab/`：

- `assets/hud04.py`：按哈希载入交付仓库中的已验证资产配置。浮动基座、29 身体关节，
  头部 yaw/pitch 结构性 fixed；参考机器人关闭碰撞和重力。
- `assets/hud04_contract.py`：纯 CPU 的模型、参考动作及版本检查。
- `tasks/shadowing/perceptive/config/hud04/__init__.py`：注册两个新任务。
- 同目录 `perceptive_shadowing_cfg.py`：继承通用感知环境，不继承 G1 环境；
  显式使用 `base_link` 根、`waist_pitch_link` 胸口、HUD04 关键连杆与末端。
- `agents/__init__.py` 和 `agents/instinct_rl_ppo_cfg.py`：独立 HUD04 runner 配置。

任务名：

```text
Instinct-Perceptive-Shadowing-HUD04-Plane-v0
Instinct-Perceptive-Shadowing-HUD04-Plane-Play-v0
```

首轮设置：2 个环境、50 Hz 控制、200 Hz 物理；动作保持运行时关节顺序，
按名称缩放并加 default offset，`last_action` 保持原始动作。
保留通用 policy/critic 观测及 8 帧本体历史，修正关节与关键连杆选择。
相机使用上游 grouped/noisy 管线的归一化和 18×32 裁剪缩放，
而不是把原始 27×48 探针输出直接交给策略。

首轮关闭质心、质量、默认姿态、执行器、材质和相机位姿随机化，以及自适应动作采样。
Plane 接触规则暂只允许足部支撑，手部不作为允许接触；地形越界终止关闭。
这些是待验证的首轮集成设置，不是已经调好的 HUD04 奖励/控制参数。
后续攀爬等手部接触技能必须重新配置接触规则。

## 安装与 CPU 检查（以下命令不启动 GPU）

在自己的工作目录准备两个 checkout。下面变量填写自己机器上的实际路径，
不是 ARIA 沙箱路径：

```sh
export HUD04_DELIVERY_ROOT=/your/workdir/ins1116
export INSTINCTLAB_ROOT=/your/workdir/InstinctLab
export HUMANOID_DESCRIPTION_DIR="$HUD04_DELIVERY_ROOT/flux/hud04_asset_camera_probe/humanoid-description"
```

原项目应处于上述固定提交；不要为切换版本而覆盖自己的未提交工作。
HUD04 模型必须是交付仓库已准备的
`HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf`，不是原始可动头部模型。

```sh
cd "$HUD04_DELIVERY_ROOT/integrations/instinctlab_hud04"
python -B -m unittest -v test_cpu.py
python -B preflight.py --check-model
python -B install_overlay.py --upstream "$INSTINCTLAB_ROOT"
# 检查上一步列出的六个新增路径后，显式应用：
python -B install_overlay.py --upstream "$INSTINCTLAB_ROOT" --apply
```

安装器检查 Git HEAD 和三个关键上游文件的 SHA256，编译新增源码，拒绝符号链接、
路径逃逸或覆盖已有不同文件。已有完全相同文件则只校验，不重写。
它不会克隆仓库、修改上游 G1 文件、运行 pip 或启动仿真。
如旧候选文件已存在且不同，应先人工检查差异；不要使用强制覆盖规避检查。

## 参考动作：默认拒绝，不能拿 G1 数据改名冒充

真实环境构造要求显式设置 `HUD04_REFERENCE_MANIFEST`。本交付**不提供伪造的已验收动作**。
使用专用数据目录，只能包含清单列出的 `*retargeted.npz` / `*retargetted.npz`，
不能混入未审查 NPZ、原始 AMASS 或对象/pickle 数组。

清单结构如下。这里故意保留 `candidate`，因此不会通过正式加载门禁：

```json
{
  "schema_version": 1,
  "robot": "HUD04",
  "root_link": "base_link",
  "quaternion_order": "wxyz",
  "angle_unit": "rad",
  "position_unit": "m",
  "model_sha256": "实际固定头部模型的SHA256",
  "status": "candidate",
  "review_evidence": {
    "constraints": "已恢复的动作与求解约束记录",
    "root_frame": "根坐标转换及单位审查记录",
    "fk_contact_replay": "FK/接触/穿地/滑动回放验收证据"
  },
  "files": [{"path": "motion_retargeted.npz", "sha256": "实际数据文件的SHA256"}]
}
```

只有完成对应审查后，才能将状态标记为 `accepted_for_plane_replay`；
这仍不是运动学门禁自动证明了动力学可行，更不是训练预算授权。
每个 NPZ 需要 `framerate`、Unicode 向量 `joint_names`、
`joint_pos[N,29]`、`base_pos_w[N,3]`、WXYZ 的 `base_quat_w[N,4]`。
加载前检查唯一关节集合、有限值、张量形状、单位四元数、位置/速度限位和文件校验和。
按关节名称重排，不裁剪、不缩放、不改动参考数据。

```sh
python -B preflight.py --reference-manifest /your/reviewed/dataset/manifest.json
export HUD04_REFERENCE_MANIFEST=/your/reviewed/dataset/manifest.json
# 明确选择仿真近似相机；不是实机标定结果：
export HUD04_CAMERA_PROFILE=sim_chest_35deg_v1
```

实机标定应使用已有 camera_mount.py 的 manual 接口，不能把头部相机位姿直接当胸口外参。
当前未恢复到可以验收的参考动作与全部历史求解约束；本次没有重新求解或处理旧动作候选。

## 下一阶段验收顺序

1. 恢复并审查 HUD04 参考动作清单，先做 CPU 检查，不新增求解预算。
2. 在兼容 GPU 环境验证真实配置导入/构造、Articulation 根路径、参考加载/重置、
   policy/critic 张量形状、相机/18×32 输入、多环境隔离和正常清理退出。
3. 新诊断应记录地面过滤足部接触、逐关节误差/力矩、根速度和姿态；
   现有 2 秒浮动探针已有漂移，不可宣称稳定站立。
4. 通过完整环境验证且确认预算后，再运行实际 PPO 更新、checkpoint 保存/重载。
   runner 默认仅 1 iteration、1 minibatch、1 learning epoch，禁止自动 resume；
   这些保守默认值仍不是执行训练授权或硬费用上限。
5. 再接 TerrainMotion 的匹配 metadata.yaml、网格和技能数据，做成功率/泛化评估。

当前不提供一键 GPU/长训练命令，防止跳过数据验收或重复提交已完成任务。
W&B 指定项目此前返回 audience 权限错误；本次没有创建新的 W&B run/artifact，
后续执行前需要恢复该项目的实际可写日志通路。