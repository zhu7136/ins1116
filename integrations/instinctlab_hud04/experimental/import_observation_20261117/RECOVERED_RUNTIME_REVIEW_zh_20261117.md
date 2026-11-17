# HUD04：失败任务真实快照已回收，但不转为通过

## 已取得的新证据

- 精确任务：`TASK_20261117_366`，实际代码 `4031da1e8d01deb132162bf8548eb8ca53a32745`，平台终态 `6`。本次为失败证据通道的单次定向短诊断；不是重跑已完成的 `732/016`，也不是训练。
- 监督器读回：子进程约 **31.20秒**，退出 `-11`；`direct_child_reaped=true`，`observation_completed=false`。本地runtime报告已存在。准确原生崩溃位置仍未知，不将最后日志行等同于崩溃位置。
- 五份完整文件：`worker_result.json`、`runtime_report.json`、`post_import.json`、`post_clone.json`、`post_physics_init.json`。通过失败专用分块通道回收，逐文件SHA256、压缩包SHA256、精确任务/提交和保守标志均复核。**回收成功不等于子进程成功。**
- 恢复后的诊断链共7个新任务：`339/341/347/350/358/359/366`；均已在2026-11-17 **14:10:56 UTC**重新读取到终态6，精确提交也匹配。账户活跃任务0、项目草稿/排队/运行任务0。见 `FINAL_RECOVERED_LIVE_STATE_20261117.json`。不要把本次最新366说成整个恢复过程只使用过一次GPU；也不要重跑其中任何任务。旧732/016未重新提交。
- 原始五份元数据在本目录的 `recovered_failure_evidence_20261117/`，逐文件哈希见 `RECOVERED_RUNTIME_INVENTORY_REVIEW_20261117.json`。没有上传原始/派生动作、密钥、带签名下载URL或完整未清洗控制台日志。
- 每次新诊断之前均有实时状态/排重检查；单任务1张GPU，子进程240秒上限、日志8MiB、零自动重试。平台计费硬截止没有获得验证，子进程上限不能冒称平台硬截止。此处是失败诊断记录，不是GPU技能训练消耗。

## 独立核对所得

用哈希与runtime报告一致的原始HUD04模型XML直接统计，没有用快照自身的计数作为预期值：

| 项目 | 三阶段共同结果 | 限定 |
|---|---|---|
| 碰撞API实体 | 43；每个link的数量均与名义URDF一致 | 不是验证PhysX cooked形状 |
| 几何表示 | 24 Cylinder、14 Sphere、3 Cube、2 Xform | 两个Xform对应左右腕网格的convexHull声明；当前快照没有其子Mesh点或真实cooked hull |
| 关节 | 29 Revolute + 12 Fixed；全部名称/类型匹配 | 头yaw/pitch均是Fixed，不进入29身体DOF |
| 自碰撞元数据 | articulation上显式 `enabledSelfCollisions=true` | 不证明具体PhysX成对过滤生效 |
| 场景坐标 | 显式 `metersPerUnit=1`、Z-up | 不是原始Step1导出单位、STL单位或对齐的证据 |
| 显式USD过滤 | filteredPairs=0、collisionGroups=0 | **不能**推断PhysX无过滤，或多环境隔离已经正确 |

快照只检查 `/World/envs/env_0/Robot`，不冒充两个环境的全量形状或跨环境隔离验收。初始化可能发生内部物理步；`explicit_physics_steps=0`不能说完全未步进。

**新的接线风险：运行时29关节的排列与URDF XML排列不同。** 已按实际名称建立可逆索引并做往返检查，见JSON的两个permutation字段。这只是名称映射证据，不是动作、reset或PPO接线已经通过；严禁按列号直接拷贝参考关节。

## CPU验证

- `python -m unittest -v test_import_worker.py test_failure_evidence.py`：41项发现，40项通过，1项跳过，无失败。见 `tests_runtime_failure_final.log`。跳过项不计作通过。
- `RECOVERED_RUNTIME_INVENTORY_REVIEW_20261117.json`由独立XML/回收文件检查生成并重新读回，确认全部三阶段的每link数量、关节名称/类型和模型哈希。不是完整几何、动力学或动作验收。
- 首次本地调用误写了不存在的测试模块名，产生一次import错误；随后以上正确命令完整重跑通过。最终测试日志对应正确命令，不把失败调用算作通过。

## 不再重复GPU取同一证据

本轮已有足够数据完成元数据核对，不追加GPU重试。失败通道不改原来的成功门禁，不吞掉非零退出，也没有生成成功证书。

下一步按依赖推进：

1. 用现有文件推进两腕网格/碰撞表示和关节名映射的CPU审查；真实cooked几何、PhysX成对过滤和跨环境隔离仍需独立、明确范围的验证。
2. 同时恢复真实Step1导出器/生成popsicle模型、root、四元数、世界/STL单位和对齐绑定；已有消费者代码与USD场景单位不能补这个来源缺口。
3. 依据材料点及多表面证据审查手足膝支撑，不能把垂向间隙、边/顶点见证或低速候选自动固化为支撑。原始997帧、90Hz和限值保持不动。
4. 来源与接触依据过关后，才连接已有35维切空间运动学、90Hz硬约束、目标地形/自碰撞和有界监督器进入求解。尚无HUD04参考动作。
5. 专用TerrainMotion配置、正确地形裁剪/边缘平移、胸口18×32真实深度、严格loader可并行准备；完整任务动态回放过关之后才讨论最小PPO链路。没有长PPO授权。

`solver_calls=0`，`ppo_updates=0`，`accepted_reference=false`，`training_ready=false`。