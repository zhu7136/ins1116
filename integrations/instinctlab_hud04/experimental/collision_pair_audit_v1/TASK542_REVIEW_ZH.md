# TASK542：独立原生回调正对照未通过

## 已执行且不可重复提交

- Flux：`PRO_20261116_016/TASK_20261118_542`，终态`6`。
- 实际代码：`be711fec11318a478be66357a0c5123721631da1`。
- 镜像：`BJX00000178/V000446`；单GPU，实际设备报告RTX4090。
- 子进程正常退出0、已回收，无看门狗触发；子进程墙钟7.432688秒，**不是平台计费时间**。
- supervisor因正对照未成立而返回失败；不是子进程崩溃，也不是PPO失败。
- 本次仅创建并启动一个新任务，无自动重试；任务132/133/138/301及固定基座任务均未重跑。

## 真实观测

| 项目 | 结果 |
|---|---|
| 配置Scene路径 | `/physicsScene`，有效且为`UsdPhysics.Scene` |
| solver Scene独立归属 | 未验证 |
| 场景单位 | 1米/单位 |
| 两个独立球体 | 各半径0.05，声明中心距0.095，动态/运动学刚体 |
| 碰撞开启、contact阈值 | USD读回true、0 |
| 接触处理禁用开关 | 初始化前true→false，reset后仍为false |
| 订阅 | 首次reset前持有双缓冲订阅holder |
| 显式物理步调用 | 8，dt=0.005s；初始化可能另含物理步 |
| 原生回调/接触点/匹配点 | 0 / 0 / 0 |
| 解析或阶段错误 | 未记录到；零事件不构成API契约兼容验证 |
| 实际material点 | 0，有效配对摩擦仍未知 |

初始化与显式步分别设置观测窗口，但两者均没有回调记录。5mm是USD声明几何的解析初始重叠，不是已验证的实时PhysX接触。未采集实时刚体状态演化或同场独立接触读数，因此不能由此断言回调接口损坏、碰撞被过滤、形状没有接触，或PhysX已实际产生接触。旧任务132的tensor/ContactSensor阳性属于不同场景及接口，不能代替本次独立物理见证。

两项软件修正中，Scene配置路径已在GPU运行中读回成立；Gf四元数序列化本次未覆盖。原生逐点material适配器仍没有实际数据可验证。child.log未包含在白名单结果中；任务日志未展示警告不等于没有child警告。

## 可复核证据

- [W&B诊断记录](https://wandb.ai/zhul136-university-of-science-and-technology-of-china/ins1116/runs/a48bd56e)，状态failed；config、summary已服务器读回。
- Artifact：`zhul136-university-of-science-and-technology-of-china/ins1116/hud04-native-contact-control-results:v0`。已读回文件清单及runtime_report字节SHA256。
- 导出ZIP SHA256：`e73d0abff86655fdb6837240628c2b763355565e32f5cafb34b27c74d97e5bd2`。5个源JSON均先验证源文件校验和，再按需脱敏；contacts含文档URL，脱敏后另有持久化校验和，不能混用。
- CPU预执行122项软件测试通过；不是122项物理测试。

## 边界和下一决策

此次服务端300秒前置例外已用于任务542，不自动延伸到另一个GPU任务。未建立平台总耗时/计费硬上限。

下一诊断设计应先补齐**实时刚体状态演化、同场独立接触见证以及实际Scene绑定/回调派发时序**，然后才能区分物理场景未成立与回调链路未投递；不要原样重提八步正对照，也不要用猜测material/friction填补缺失数据。

无HUD04机器人、相机、动作、优化器或PPO参与本次试验。`accepted_reference=false`，`training_ready=false`。原候选4692 strict/277 unresolved等历史约束未重新测量、更未解决。