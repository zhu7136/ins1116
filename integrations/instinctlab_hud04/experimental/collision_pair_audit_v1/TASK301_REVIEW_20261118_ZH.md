# TASK_20261118_301 复核与元数据采集修复

## 复用既有 GPU 结果，不重复提交

- Flux 项目 `PRO_20261116_016`；任务 `TASK_20261118_301`，服务器回读状态 `5`。
- 部署提交 `1f112e121e28018b63ad3a5146e36be4ed9145be`。
- W&B：<https://wandb.ai/zhul136-university-of-science-and-technology-of-china/ins1116/runs/ys4k6zk2>。
- 既有结果 Artifact：`zhul136-university-of-science-and-technology-of-china/ins1116/hud04-runtime-observation-results:v0`。
- 本次从终态任务官方导出的日志中恢复 allowlist JSON，先校验 ZIP 与四个源文件 SHA256，再按需脱敏；未创建新 W&B run/artifact，未启动、重试或停止 GPU 任务。
- ZIP SHA256：`8093d6fa40628ab37fe155b02829cc92e9b253d634a68de22cd1083e561dd3be`，与既有 W&B run 汇总一致。
- `contacts.json` 的文档 URL 被通用恢复工具脱敏，因此保存的是 `contacts.sanitized.json`，不能把脱敏文件的 SHA256 冒充原文件 SHA256；恢复收据分别记录两者。其余三个文件持久化字节与已核验的原文件相同。

## 已验证与未验证

- 新转换的浮动基座运行时，29 个身体关节、头部策略 DOF 为 0；不是历史 USD 字节恢复。
- 8 个显式物理步，dt=0.005 s；初始化可能额外步进，不把 8 作为总物理步数。
- worker 返回码 0，子进程约 17.46 秒回收；这不是服务端硬终止或总计费上限证明。
- 接触回调 0、接触点 0；接触报告投递、原生 buffer 健康、最终 solver filter、cooked shape 等价性仍未验证。
- 未添加地面/地形，未使用参考动作，未测试完整任务，求解器与 PPO 更新数均为 0；不能报告站立通过、足部支撑通过或参考动作通过。
- 两个 wrist roll joint 的组合 USD 元数据均读到 `physics:collisionEnabled=false`（schema 默认，非 authored），两侧刚体 report threshold 读到 authored 0。这些是组合 USD 读数，不是最终碰撞过滤原因或回调通道有效性证明。
- `accepted_reference=false`、`training_ready=false`。不豁免参考动作碰撞对，不猜测有效摩擦，不以 CPU 初始插值诊断代替受约束参考重建。

## 发现并修复的观测缺口（CPU 验证，尚未 GPU 验证）

1. 原脚本硬编码 `/World/physicsScene`，本次返回 `Exact prim unavailable`。改为被动读取调用方 `sim.cfg.physics_prim_path`；配置缺失/非法时明确记录 unknown，不猜路径、不自动回退。读取路径字符串仍不证明 prim 存在、类型为 Scene 或它拥有当前 solver。
2. 元数据原先尝试把 `Gf.Quatd/Quatf` 当 iterable，导致 root/link 朝向、惯性主轴等属性未被读出。现在由 GPU host 显式传入 `Gf.Quatd/Quatf/Quath` 类型，通过公开 getter 保存带标签的 wxyz 四元数，不归一化、不把 USD 默认值当当前仿真状态。
3. 元数据只读路径验证允许 `/physicsScene`、`/Looks/...` 等非 `/World` 命名空间，仍要求 canonical exact prim path、最多 32 个。actor/collider registry 原有 `/World/` 限制不变，未放宽碰撞对匹配。

没有在本 CPU sandbox 安装/加载 pxr 或 Isaac，也未执行 runtime_trial 的 `--execute`/`--child`。新增 10 项 fake-object/AST 软件测试；碰撞组件共 **113 项**、全局轨迹组件 **50 项**全部通过。实际 SDK 类型、`sim.cfg` 字段、有效 Scene 及接触投递仍需以后独立的短时运行验证。本次不把这些软件修复写成已通过 GPU 验证。

上述完整测试套件沿用部分本地 `/workspace/InstinctLab` 源码 fixture，不宣称新 clone 即可无外部依赖运行；新加的 10 项回归自身使用 fake objects/AST，不依赖 Isaac 安装。

历史 `RUNTIME_PREPARATION_MANIFEST.json` 与 CPU checkpoint manifest 保留为对应提交的历史证据，不是本补丁的部署/运行清单。运行前应基于实际部署提交重新产生并核验 preparation，而不是重放旧创建/启动脚本。

## 后续判据

先验证 Scene 身份、采集完整性和接触投递的正对照，再收集真实接触两侧的材质/形状绑定，之后才能推进有效摩擦与参考动作约束。正对照须标记为独立诊断，不能偷偷改变原验收模型参数，也不能用“无回调”替代无碰撞证明。

必要短时 GPU 的常驻授权已经记录，不需逐次申请；这不自动扩张为无限预算、长 PPO 或取消此前的服务端 300 秒硬限时条件。旧任务采用 worker watchdog 且 `platform_hard_cap_verified=false`，其成功不能反向证明该条件已满足；本次未作新的 GPU 提交。