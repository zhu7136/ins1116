# HUD04：从预检进入实现与首次时序求解（2026-11-17）

## 本轮实际交付

基线远端 HEAD `2e0506721c1cae6a9bd117899db1b3e48192b0ba`，恢复并核对后在独立目录实现；未重置旧工作树，未执行旧 GPU 创建/启动/发布脚本。

### 1. 全长关节时序求解已运行，不是完整重定向

- CPU 沙箱，HiGHS 线性规划，29 个独立关节子问题，每个联合求解全部 997 帧。
- 目标最小化与源关节序列的 L1 偏差；位置上下限、相邻帧 `abs(q[t+1]-q[t])*90<=vmax` 为硬约束。未放宽模型限值，未事后裁剪输出。
- 完整保留 997 帧、90 Hz、29 身体通道；未输出任何头部动作。
- 源动作 SHA256 `56158e8679580910ccff375963d025b138636acfe70f5b1f88381837209b2056`；目标模型 SHA256 `865a4344e2f838d939c2549ec0d9e771dc6c2eb745670d96bba78e162d2880d7`，求解前后复核不变。
- 原序列 261 个位置超限帧、165 个超速区间；输出两项均为 0（检查容差 1e-6）。最大位置残差 0 rad，最大速度残差约 3.55e-15 rad/s。
- 求解耗时约 0.476 秒；子进程正常退出并回收；父进程 90 秒截止、各 LP 最多 2 秒、无自动重试。
- RMS 改变量 0.038114 rad；最大改变量 0.835342 rad（47.8616 度），发生在源第 608 帧（0-based）的 left_knee_joint：-0.0118445 -> 0.823498 rad。
- 该明显变化说明仅满足关节约束不足以保留支撑/动作语义；不得据此开始训练。
- `step1.joint_fit.npz` 故意没有 root/四元数字段，也不使用 `retargeted.npz` 后缀。它只是优化中间量，不是 HUD04 参考动作。
- 真实执行：`solver_calls=29`（内部 LP 数），`full_retarget_solver_calls=0`。没有根位姿、地形、自碰撞、接触约束；不能把此次结果称为完成了有界全身重定向。

### 2. 严格 loader 已接入源代码

- 新增 CPU 严格读取器 `overlay/instinctlab/assets/hud04_motion.py`。
- 新增实际 Amass/TerrainMotion 子类和 cfg `overlay/instinctlab/motion_reference/motion_files/hud04_motion.py`，覆盖文件分派和读取，不再走上游 `allow_pickle=True` 分支。
- Plane 路径现在使用严格子类；TerrainMotion 使用同一个读取实现。
- 按实际运行时关节名称重排，拒绝重复、缺失、多余和头部名称；保留 90 Hz、997 帧；拒绝对象数组、非有限值、非法四元数模长、位置/速度超限以及重采样/裁剪配置。
- 合成测试实际调用 adapter 的文件分派/读取/打包入口；仍不是 Isaac/PhysX 实例化证明。

### 3. 专用 TerrainMotion 配置已实现，尚未在 Isaac 进程实例化

- 新增 Terrain 与 Terrain-Play 注册，不继承 G1 配置，不加载 G1 权重。
- 复用 HUD04 资产、头部固定、29 动作、胸口相机与原始 18x32 深度预处理。
- 恢复上游 motion-matched 地形生成器（9x12 裁剪、边界高度与共享平移逻辑），不使用 Plane 或原始 STL 绕过实现。
- 2x2 子地形、2 个环境，使用严格 TerrainMotion；保持原始参考采样，不插值或跳帧。
- metadata、地形、支撑区间通过 manifest 文件路径和 SHA256 绑定。当前仅接受 Step1/整数 terrain_id=0，规避上游 terrain ID 实现与通用文档之间的差异。
- 六个手/足/膝 link 必须全部显式声明审核区间；空列表意味着从不允许。未填写、未审核或不匹配的数据直接拒绝；未生成任何真实动作支撑标签。
- 条件接触项按实际 buffer reset stub + manager 参考时间映射到 90Hz source frame，不使用 episode step 充当源帧，也不把历史接触力配给当前帧掩码。
- 无任何全局手/膝豁免；其他 link 始终不允许支撑。接触力阈值 1N/500N 为原有诊断默认值，绝不是已验收的安全、穿透、滑移或跟踪标准。
- Terrain PPO runner 默认仅 24 步、1 update、1 epoch、1 minibatch、不恢复权重。没有执行 PPO，也没有因此获得长训练权限。
- 安装器更新为显式 10 文件集合；仍拒绝覆盖已存在的不同代码。已有旧 overlay 请使用新的独立 pinned InstinctLab checkout，不要删除/覆盖旧树来强行安装。

## 测试与日志

- `IMPLEMENTATION_TESTS.log`：25 个新增 CPU/unit 测试全部通过（2.704 秒）。包括真实 LP 输出残差独立复核、时序耦合非逐点裁剪、实际 adapter 命名映射、reset-stub 时钟、条件支撑掩码和拒绝路径。
- `LEGACY_CPU_REGRESSION.log`：37 个旧 CPU 回归测试全部通过（0.467 秒）。仅适配 10 文件/4 注册、新方法参数和显式 delivery 路径；保留原资产/参考/禁止覆盖等测试。
- 共 62 个测试通过，无跳过；没有声称重新运行此前全部测试套件。
- 测试中出现的 accepted 状态只属于临时合成 schema fixture，不是对真实动作的验收；临时文件不发布为训练数据。
- 未在真实 Isaac 进程构建 Terrain 配置，未做动态回放/reset/训练/深度帧运行验证。原 native `-11` 退出故障未在本轮修复或重试。

## W&B 与 GPU

- 在线 `wandb.init` 实测仍返回 `invalid audience for request`；最初轻量 `Api.project` 返回非空对象不代表运行写权限恢复。
- CPU 求解以显式 offline W&B run `w8qv59ak` 记录，尚未同步，**没有可用的在线 run URL，也没有服务器读回成功声明**。本仓库中的结果/日志/候选是当前可取回交付。
- 本轮 GPU 提交 0，PPO update 0；没有查询后声称账户当前活跃任务为零。任何以后新 GPU 提交前仍须实时查重，不得重交旧 732/016/339/341/347/350/358/359/366。

## 下一步：把第一层解接入全身求解，而不是重新取证

1. 把已求得的全长关节时序作为优化中间量，与根位姿/躯干/手足膝几何跟踪共同求解；不能直接复制源 root 当成已标定的 HUD04 root。
2. 对 root/quaternion/world-STL 对齐缺口做有限、有记录的重建/标定决策；不得用最低穿透分数暗中选择解释。诊断候选必须明确条件，不等同于 accepted reference。
3. 将审核后的支撑区间、实际裁剪地形含侧壁以及自碰撞约束加入求解；先输出真实候选与失败/残差，再决定是否具备短动态回放条件。
4. 如需 GPU，仅做与新实现对应的短动态/reset/深度检查，提前实时排重；不重复导入资产诊断。最小 PPO 链路检查仍未执行。

当前：`accepted_reference=false`，`training_ready=false`。原始文件与模型限值不变。

## 复现说明

主交付是 top-level `integrations/instinctlab_hud04/overlay` 与本目录代码。实验脚本不会自动安装或启动 GPU。

新环境需设置 `HUD04_DELIVERY_ROOT` 为 ins1116 完整 checkout、`HUD04_STEP1_MOTION` 为原始完整 Step1 NPZ 路径。`run_joint_fit.py` 默认尝试在线记录；已知 audience 问题时可显式选择 `--offline-after-audience-failure`。先查看 `JOINT_FIT_INTENT.json` 和既有产物；脚本遇到已有 intent 会停止，不能盲目重跑覆盖。

`python -m unittest -v test_implementation` 只运行 CPU 测试。完整 Isaac 任务还需要真实已审核的 terrain manifest、相机 profile 和依赖环境；不能把本目录的 joint_fit 文件改名来绕过这些要求。