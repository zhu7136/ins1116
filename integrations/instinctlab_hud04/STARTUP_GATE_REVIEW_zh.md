 # HUD04 正式启动门禁接线复核 — 2026-11-18

## 结论与边界

已把接口检查接到审核版本的正式 `scripts/instinct_rl/train.py`，不是让调用者
自行记得执行独立 hook。安装时默认包含，HUD04 启动时默认检查后退出。
本次完成的是**源码接线、CPU 正反例及实际安装/入口拒绝验证**，不是完整任务
GPU reset 成功、动力学验收或 PPO 训练。`training_ready=false`。

交付分支保留并行提交 `70393c676c85888db0bdd92f8ab7e64fe83fa88d` 的全部审计文件；
该提交新增的方向拟合非收敛/姿态可观测性说明，不会把诊断升级为获验收参考动作。
本轮没有重新提交 Flux 任务，没有调用求解器，没有 PPO 更新，没有新增 W&B run。

## 固定输入与安装行为

- 上游 InstinctLab：`de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057`。
- 审核的原始 `scripts/instinct_rl/train.py` SHA256：
  `8a38599e47ce94940d6488c89981c4ac1dd601e400ea5cb96e8b67d1ffdeaeb2`。
  `fixtures/upstream_train.py.txt` 是该版本原始文件的逐字节测试副本，保留原版权头；
  测试会验证哈希并执行真实补丁入口/函数体，不另写一个理想化入口冒充它。
- 原 overlay 的 10 个文件继续按新增规则处理；新增两个门禁 helper 和安装 receipt。
- **唯一上游源码替换是 train.py**：只接受完整原始字节或完整、相同的本版补丁。
  不是任意字符串替换，不接收部分补丁，不覆盖已有不同 helper 或 receipt。
- 安装先检查所有计划目标、源码身份和符号链接，再写入；结束后逐文件回读并再次
  验证完整计划为 no-op。不能据此声称文件系统事务原子性；中断后不应盲目重试或
  放宽身份检查，应检查具体目标，必要时使用新的干净检出。
- receipt 包含被补丁入口、helper 和全部 overlay 的哈希；启动前和 reset 后复查。
  这是版本一致性检查，不是签名、恶意本地管理员防护或物理验收证明。

## 必经执行顺序

1. `train.py` 导入 Isaac/AppLauncher **之前**，执行 `prelaunch(argv, __file__)`。
   非 HUD04 任务不执行 HUD04 检查；HUD04 只支持精确 Plane/Terrain 非 Play 名称。
   禁止参数缩写、重复 task、环境数不是 2、Hydra `key=value` 覆盖、分布式、
   `LOCAL_RANK`、视频/debug。未知 HUD04 模式/任务必须拒绝，不降级 G1。
2. 检查安装身份；在不执行父包 `__init__` 的条件下解析九个选定源码文件并核对
   既有哈希，包括实际使用的 wrapper/motion/normalizer 深度链依赖。
3. 执行 CPU 输入预检：固定头部 29 身体模型、参考 manifest 的已声明验收状态和
   模型/文件校验、严格按名 **997 帧 / 90 Hz** 加载器、相机配置；Terrain 还须通过
   原有地形/支撑 bundle 检查。无参考不回退静态站姿，无 G1/候选文件兜底。
   baseline 资产文件**只读取字节核对哈希**，不在 AppLauncher 前执行它；控制契约和
   相机 mount 模块为 CPU-only。测试显式禁止该阶段导入 Isaac/Omni/PXR。
4. 通过前述检查才允许 AppLauncher 启动、正常任务配置解析及完整环境构造。
5. 正常 wrapper 构造会 reset，但丢弃其返回值；门禁再调用一次公开 `reset()`，
   取得实际扁平观测。不调用会推进历史的额外 `compute()` 冒充 reset 观测。
6. 再次核对文件/模型/参考/源码路径身份；确认实际 config 类、motion loader、
   参考根目录、无插值重采样/跳帧及无隐藏动作裁剪。调用现有 live hook 核对：
   29 身体动作/参考/关节观测解析顺序，scale/default offset、policy 1990 维、
   18×32 深度、8 帧本体历史及批次有限性。头部两个自由度不进入动作空间。
7. live hook 必须明确返回通过；随后独立 `require_runner_permission()`。
   两者都位于 runner **构造**、checkpoint 加载、learn 前，失败直接传播。
   默认 `HUD04_STARTUP_MODE=check-only` 通过后退出 0，**不构造 runner**。
   未来显式 `train` 仍受前述检查约束；此开关不是训练就绪或预算批准。
8. HUD04 main 用 `finally` 清理环境/仿真。环境关闭幂等；清理失败不得把 check-only
   伪装成功。最终清除 `SimulationContext`，再 `app.close(wait_for_replicator=False)`。
   本次只验证 CPU 替身中的顺序，未重跑旧 GPU 退出诊断。

安装器没有关闭门禁的开关。这里不覆盖其他自定义入口、直接 `gym.make()` 或评估/
Play 脚本，也不是无法被人为修改源码绕过的安全沙箱；“必经”范围是本安装器生成
且通过 receipt 检查的上述正式训练入口。

## 本次实测

### 定向 CPU：132/132

```sh
cd "$HUD04_DELIVERY_ROOT/integrations/instinctlab_hud04"
python -B -m unittest -v test_startup_gate.py test_runtime_gate.py test_cpu.py
```

62 项启动门禁测试 + 33 项运行时接口测试 + 37 项既有 CPU 测试。
覆盖安装计划不写入、默认安装、重复安装 no-op、修改/缺失/符号链接拒绝、CLI 绕过
拒绝、真实 CPU 输入加载、缺少/候选/短参考拒绝、相机/地形拒绝，以及实际 patched
main 在成功 check-only、接口失败、reset/关闭异常时相对 runner 的执行顺序。
模拟 `train` 正例只验证权限/接线，未真正构造训练 runner 或运行 PPO。
测试用的 accepted manifest 声明、运动数组、环境及观测均为临时**合成软件夹具**，
不发布为获验收机器人参考动作，不证明物理可行。

### 扩大回归：257 项，250 通过、7 error

```sh
python -B -m unittest discover -s . -p 'test_*.py'
```

7 项均在既有 `test_terrain_surface_geometry` 的 `CropTests`（3 项）及
`IndependentKernelTests`（4 项）导入几何处理/核验器时缺少 `trimesh`；未屏蔽、
跳过或算作通过。未在线安装依赖。不能称全套绿灯。

### 真实固定检出安装与正式入口负例

在独立干净上游 worktree 安装，而不是修改旧 probe 工作树：

```sh
python -B install_overlay.py --upstream "$INSTINCTLAB_ROOT"   # 只检查计划
python -B install_overlay.py --upstream "$INSTINCTLAB_ROOT" --apply
python -B install_overlay.py --upstream "$INSTINCTLAB_ROOT" --apply
```

最终检查 worktree：`/workspace/InstinctLab-hud04-startup-final-check`。
第一次结果 `startup_gate_default_on=true, source_overwrites=1`；第二次
`source_overwrites=0`。全部安装文件按 receipt 回读一致。

实际执行安装后的正式入口（不是 CPU stub）传入 HUD04 Plane 名称，因本 CPU 环境
未安装/注册 IsaacLab 源码而以 **exit 1** 拒绝，报
`Cannot resolve source before AppLauncher: isaaclab`。拒绝发生在 AppLauncher 之前，
未启动 GPU。这证明实际入口负路径，不证明完整环境的成功路径。

## 后续不能跳过的条件

先补齐同源根轨迹/坐标与支撑证据并形成通过既有验收的 HUD04 参考文件；不得将
方向拟合局部诊断、非收敛输出或临时合成 fixture 填成 accepted。准备固定版本
IsaacLab/InstinctLab 环境和真实模型输入后，先用**默认 check-only** 验证真实完整
任务 reset/接口成功与失败路径；在此之前不得声称门禁已在目标 GPU 镜像验证。
物理接触/稳定性与预算边界仍需单独满足，不能因启动门禁的软件测试通过启动长训练。