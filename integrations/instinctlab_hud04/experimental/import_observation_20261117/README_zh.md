# HUD04 真实导入观测短测：两次已终止，未取得完整证据（2026-11-17）

## 本阶段实质增量

新增独立 `worker.py` 和 `import_observation_probe.py`，接入已经发布的只读 USD 采集器。初版只做CPU准备；后续已执行首次新GPU导入短测，失败原因与修复记录见下文。**未运行重定向或 PPO，默认命令仍只打印计划，不导入 Isaac。**

真实执行设计为 2 个环境、`replicate_physics=True`、请求场景碰撞过滤、关闭相机；新建独立场景读取原固定头浮动基座 HUD04 资产配置。源资产 manifest 全量校验，额外锁定 URDF 与 asset_config SHA256；实际 Git HEAD 必须匹配显式指定值且所有 tracked 文件无修改。不得通过 reset 清除检查失败。

使用 `InteractiveScene._add_entities_from_cfg` 的观测子类钩子，分别采集 `post_import`、`post_clone`、`post_physics_init`。钩子不存在/未触发则失败，不伪造备用阶段；运行时记录 InteractiveScene 源码哈希，不把参考源码版本当作镜像真实版本。初始化后核查真实关节名称数量为29、无头部关节、浮动基座。没有动作管理器、运动数据加载或策略更新。

这是**独立导入/克隆/初始化探针**，不是实际感知任务实例，不含原始 Terrain，也不验证站立、手足膝接触、跨环境动态隔离、PhysX cooked 顶点或实际 shape-pair 策略。USD 采集器内 `gpu_started=false` 表示采集器自身不启动GPU；新增 `runtime_execution=true` 表示调用场景来自真实执行（当前尚未产生此类文件）。阶段报告的 `actual_physx_pair_filters_verified` 和 `cooked_geometry_verified` 始终 false。导入或快照成功不等于碰撞验证通过。

## 运行与清理边界

`bounded_import_runner.py` 与原 CPU-only supervisor 分离，避免原监督器强制隐藏 CUDA 的设置。新监督器只运行可信子进程：240秒子进程墙钟截止（库上限300秒）、日志8MiB、零重试；只按自己创建的进程组回收，无进程扫描、无守护接口。清理宽限最多0.8秒；最后正常退出且三阶段报告齐全才记 observation_completed。报告先写完但进程卡住、异常、非零退出、日志超限、任何接受标记被抬高均失败。

父监督器通过PIPE+selectors限制日志流8MiB，**不再施加进程全局RLIMIT_FSIZE**。旧实现的RLIMIT_FSIZE误将USD转换资产截为8MiB，导致首次新GPU短测 TASK_20261117_339 失败（24秒平台起止时间差；不是计费上限或费用证据）。实际USD日志显示预期table offset25896964而文件仅8388608，不能将其视为模型几何问题。该任务已终止，不可原样重跑；修复后需新提交、新排重、新任务。转换TMPDIR改到本次新输出目录内，不修改原模型文件。26MiB合成转换文件回归验证其不再受日志上限影响，日志洪泛仍失败停止。

这不是任意代码安全沙箱。child.log 为本地诊断，不经完整证据回传；失败入口允许输出脱敏的末尾诊断片段以定位错误。允许回传的5个JSON以固定清单ZIP编码，展开总量≤1MiB、压缩≤512KiB，8192字符分块，文件与归档SHA256可核对；缺文件/失败结果拒绝回传为完整观测。无自动重试/任务删除/费用推断。

**子进程时限不是 Flux 云任务/计费硬上限。** 2026-11-17T12:50:17Z 的官方能力描述只明确 `task.stop` 对 running 有效；batch.stop 跳过非运行；未发现覆盖 preparing/queued 的可靠取消或创建参数硬截止。runtimeReminderConfig 不是硬终止。不要猜测额外字段或用删除当取消。真实新GPU提交前仍须重新查实时任务、排除已完成任务重复、核对镜像/项目和可控执行/费用边界。

## 入口（默认仍是 plan；显式执行记录见下）

```bash
python -B worker.py
```

只有在新任务的执行边界成立之后，才显式加入 `--execute --repo-root <经核验且干净的实际checkout> --expected-commit <该checkout完整提交> --output <不存在的新目录>`；可用 `--isaac-launcher` 指定实际存在的本机 launcher。未指定时仅在当前解释器确实可发现 isaaclab 或历史候选路径确实存在时尝试；文件存在不等于运行验证通过。新worker不创建或启动Flux任务，不运行旧worker，不安装依赖。

源资产读取的是已有配置模块，不是旧发布/创建/启动脚本；manifest、29身体/0头部动作、模型限值不改。没有重写原始数据、90Hz、参考动作或验收清单。

## 已实际验证（CPU）

`HUD04_TEST_REPO_ROOT=/workspace/hud04-recovery-bf4a057 python -B -m unittest -v test_import_worker`：修复后15项通过，含真实已恢复资产的离线全manifest/SHA256/固定头结构检查、26MiB大文件/8MiB日志上限分离；其余子进程是明确标记的 synthetic CPU fixture，不是HUD04物理结果。测试输出目录唯一新建、无覆盖或删除。未设置 HUD04_TEST_REPO_ROOT 时该离线资产项会显式skip，不将skip当通过。

初版两入口 plan 实际运行，无 Isaac 导入/GPU启动。后续用户授权下首次新GPU任务TASK_20261117_339的App已启动，但因上述全局文件限制未取得有效快照；终态6已读回，无自动重试。真实结果须以每个新任务独立证据为准。现阶段 `solver_calls=0`、`ppo_updates=0`、`accepted_reference=false`、`training_ready=false`。导出/根坐标/四元数/世界单位和支撑时序来源仍待证实，不用更小间隙替代来源证明；两处腕部名义重叠仍保留，不自动豁免或判实际违反。

## 两次新GPU短测已收尾，未通过运行时观测验收

- TASK_20261117_339：实际提交0acf47c0；24秒平台起止差，因监督器全局文件限制导致USD转换文件截断。终态6已回读。
- TASK_20261117_341：实际提交6f0f7e11；修正日志流限制后执行独立新短测，子进程在240.519秒触发child_wall_deadline，监督器回收直接子进程，完整runtime_report与三阶段快照未回传，平台终态6已回读。return_code=0不是成功证据，超时/缺完整报告门禁已正确拒绝观测完成。251秒平台起止差含环境/收尾，不是240秒计费上限证明。
- 后者日志已进入URDF导入警告（约14秒），但没有足够阶段标记证明停在何函数；不能从shadercache/kvdb/GLFW警告单独确定根因，也不能断言修复已在完整资产导入中验证。未修改原模型、动作、90Hz或限值，无重定向/PPO/站立验收。
- 现在只补充更细的phase标记和60秒周期自有子进程Python调用栈（不记录变量值、无/proc/宿主/其他任务访问），继续受同一8MiB日志预算及240秒终止机制约束。**本instrumented版本未提交第三次GPU任务**。追加两项phase门禁测试后17项CPU测试通过；不是GPU物理通过。
- 结构化证据摘要见同目录 `GPU_IMPORT_SHORT_TEST_REVIEW_20261117.json`，包含各任务代码、平台起止、终态、日志SHA256和未证实事项，不发布原始日志/签名URL/凭据。再次GPU前仍须实时排重；不能重放这两次创建/启动脚本，也不能清共享缓存或盲目扩大预算。