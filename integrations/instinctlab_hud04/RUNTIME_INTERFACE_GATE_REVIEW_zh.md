# HUD04 真实环境接口门禁：CPU 实现与边界

## 恢复基线

本轮开始时本地旧工作树停在 `8b421f55`，但只读检查发现远端已到
`f4e25561c4c8655faa73b7f68086a61a6034f1de`。因此从最新远端建立独立工作树，
没有覆盖旧工作树、重做已有 loader、重复任何 GPU 任务或重跑求解。

已读最新方向拟合说明、严格加载器、真实失败快照复核和训练关键路径。
发布前远端又前进到 `805e620d195bbdd6794009ba368f73890c8af927`，已读新增全长
诊断说明并仅做 fast-forward；新增修改仅在 experimental/full_direction_20261118，
没有覆盖本轮门禁文件。保留该并行提交，新增门禁提交以它为父提交。
最新远端结果已覆盖997帧的骨盆局部关节方向诊断，但不含完整世界根轨迹，
未通过参考动作验收；不要继续误称仅完成三个局部窗口。
已回收的运行时快照说明真实关节顺序不同于 URDF 列顺序；快照回收成功不能改变
原任务非零退出、接触/物理验收未完成或 `training_ready=false` 的事实。

## 新增的实际能力

`runtime_gate.py` 提供可供后续完整任务诊断显式调用的只读 hook：
`check_live_environment(env, wrapper, observations, names, control_values)`。

- 不导入 IsaacLab/Isaac Sim、不构造环境/runner，不调用 reset、compute、step 或求解器。
  必须传入**正常参考动作验收路径**构造并 reset 后已有的环境和展平观测。
- 核对九个**已经加载**的选定 Python 模块源文件 SHA256；不是凭 README 的版本号
  推断 Flux 镜像依赖。缺模块、无可读 `.py`、字节不符均拒绝。
  所选 IsaacLab 源来自先前审计的 `f73c33173801f5f8afea4142482e47b7710c2b75`，
  InstinctLab 来自 `de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057`。
  **并不覆盖所有 Python 文件、Instinct-RL、原生二进制、monkeypatch 或 PhysX。**
  如实际镜像不同，应先保存差异审查，不能为通过门禁自动重写预期哈希。
- 核对恰好两个环境、29 个唯一身体关节，头部不进入动作或关节观测。
- 从 action manager 的已解析关节 ID、motion reference manager 的关节名、
  observation manager 已解析的 joint_pos / joint_vel 关节 ID 读取顺序，要求与
  articulation 的实际运行时顺序完全一致。不会自动调换数组掩盖错误。
- 核对唯一 `joint_pos` 动作、运行时逐名称 scale、default offset、无新增 clipping。
- 核对 policy 观测项顺序、10 帧参考、8 帧本体历史和深度 `(1,18,32)`，
  对应展平宽度 **1990**；核对实际 policy/critic 批次宽度和有限值。
- 成功返回仍显式 `reference_acceptance_proven_by_gate=false`、`training_ready=false`。

这里使用了经字节绑定的 `_joint_ids`、`_scale`、`_offset` 和
`_group_obs_term_cfgs` 私有字段，以核对**解析后**的真实接口；不是宣称它们为公共 SDK。
字段变化必须失败并重新审查，不能回退为只比较未解析的正则表达式。

## CPU 验证

```sh
cd integrations/instinctlab_hud04
python -B -m unittest -v test_runtime_gate.py
```

合成夹具使用真实 HUD04 关节名称与复用的逐关节控制规则，但不是真实环境数据。
测试包括调序、头部/重复关节、动作维度、scale/offset、裁剪、历史、深度形状、
非有限观测、源码篡改拒绝，以及假 manager 的 hook 提取测试。
假 manager 故意没有 reset/compute/step 方法，验证 hook 不需要调用这些接口。
fixture 模块名和测试哈希替身只用于 CPU 软件测试，不能作为真实源码身份或 GPU 证据。

本轮实际结果：新增 **33 项** CPU 测试通过；与 `test_cpu.py` 一起定向运行
共 **70 项通过**。扩大到本目录 `test_*.py` 共发现/运行 195 项，其中 188 项
通过、7 项报错；7 项均因本轮沙箱缺少 `trimesh`，涉及既有地形裁剪与独立距离
模块导入，没有静默跳过、伪装通过或在线安装依赖。完整回归尚未全绿。
没有把先前 55 项方向求解测试等历史结果计入本轮实测数量。

现有模型/基础测试使用原已验证工作树中的固定头部 URDF 与基线控制模块；
新 worktree 本身不自动复制或重新下载模型目录。依赖缺口不改变当前软件门禁的
定向测试通过事实，也不能据此宣称物理或训练验收。

## 没有声称完成的工作

**这个 hook 尚未接入上游 train.py，也没有修改既有任务注册或参考动作门禁。**
它不是新的训练启动器、全局拦截器或训练权限控制，不能阻止用户直接运行上游脚本。
发布的是“运行时接口诊断组件”，不是“完整训练入口已经 fail-closed”。
下一次实现/运行完整任务诊断时，应在正常环境 reset 后、创建 runner 前显式调用；
异常必须让诊断失败，不得吞掉警告继续训练。

本轮没有实际构造 IsaacLab 环境，没有真实相机帧/接触/复位/多环境隔离验证，
没有 GPU 提交、没有新的实时 GPU 状态查询、没有优化器调用和 PPO 更新。
旧任务终态记录不是新的实时排重凭证，后续任何新 GPU 提交仍须先查询服务器。
深度形状正确也不能证明深度时钟在走、噪声/归一化数值正确或策略使用了图像。

后续顺序不变：先闭合参考动作来源、世界根与地形/支撑依据，保留原始
997 帧 / 90 Hz、模型位置/速度限值、头部零位；再做含真实接触/相机的完整任务
动态回放与失败清理验收，之后才讨论有界的单次 PPO 链路验证。
不因软件门禁测试通过而追加 GPU 或启动长训练。