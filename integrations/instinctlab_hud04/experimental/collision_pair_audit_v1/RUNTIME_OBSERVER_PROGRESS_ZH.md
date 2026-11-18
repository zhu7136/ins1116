# 恢复GitHub授权后的实现进展（CPU，未提交GPU）

## 授权与交付

2026-11-18本轮重新检查Secrets，`ins1116`恢复可用；只将它用于GitHub。上一轮7文件审计包保持原字节不改，其W&B下载哈希仍可比对。本次将这些文件与本轮观测器、测试和说明一起发布到既有分支；没有重放旧发布/任务脚本，没有更改原碰撞模型、过滤、参考动作或验收策略。

## 已实现，不只是建议

`native_contact_observer.py`是**已有仿真进程内的只读原生事件接收器**，没有launcher、订阅操作、仿真步进或任何碰撞属性写操作。

- 从已审计JSON生成两对精确actor/collider路径登记，不按关节邻接、名称后缀或Mesh子路径猜测有效配对。
- 接收现有运行时适配层提供的header/contact数据与路径解码函数，绑定actor0/actor1/collider0/collider1，支持事件顺序反转。
- 明确区分“USD碰撞prim路径绑定”与“PhysX cooked shape-ID/顶点等价”。没有后者就不声明验证完成。
- 保存原生position/normal/impulse/separation字段；字段缺失保留null。separation只标raw，未绑定版本语义与物理单位前不换算为穿透深度。
- 默认最多256个事件、4096个接触点；越界、缺失形状字段、非有限值、无法解码的ID均标失败，绝不截断后报成功。无关/未匹配事件仍保留，不视为被豁免。
- 回调捕获错误但不向用户仿真循环抛出异常，不修改输入buffer；报告为深拷贝并可计算确定性摘要。
- 没收到事件不等于过滤；没有订阅成功和仿真时间覆盖回执，就始终保留coverage=false。actual_pair_filter_cause_verified、cooked_equivalence、physical_penetration_depth_verified和training_ready始终false。

**边界：该接收器通过的是合成header/point对象的CPU单元测试，尚未在当前Isaac/PhysX构建中接入。** 它是可测试的采集逻辑，不是原生API可用性或物理结论。Python原生header字段、decoder、订阅生命周期等必须用目标镜像版本的公开接口逐项绑定，不通过假定方法名来补证据。

## API证据边界

本轮只读获取并检查IsaacLab v2.3.2官方contact_sensor.py固定Git blob：

- 仓库 `isaac-sim/IsaacLab`，blob `2a85a2661f6acdfd96ba18745a688f714b8c6752`；
- 该版本第400–405行使用RigidContactView.get_contact_data()的6项返回，仅直接采用接触位置及count/start索引；
- 第467行链接到原生RigidContactView文档；**该调用并不能证明有逐shape-ID数据，也不能仅凭返回项位置认定第4项是物理穿透量**。
- 历史探针只保存body-pair力/点数，没有本次原生事件header的shape路径或有符号分离记录，不能离线补造。

## 下一步的最小接入契约（尚未执行）

1. **能力绑定先行**：目标镜像中读取真实IsaacSim/PhysX/API版本；从该版本公开接口绑定订阅、路径解码、事件类型、buffer索引、separation符号/单位。能力缺失即输出UNRESOLVED，不改用body-pair计数冒充。
2. **接入已有原articulation**：用当前已导入的原始HUD04、原惯性/根/关节状态，不生成独立kinematic父体副本。读回所有碰撞形状开关、关节过滤属性、接触处理设置、完整actor/collider对应与状态签名；同时记录时间覆盖、API有效性和正对照。
3. **cooked独立门禁**：若公开API能读cooked形状/顶点/局部变换，则读取并绑定实际shape身份；若不能就保留unknown。USD mesh、convexHull声明、路径匹配和接触法线均不能代替这一步。
4. **仅在另行允许因果试验时**，比较同一原articulation的指定jointCollision关闭/开启/恢复状态；所有其它状态签名一致，并确认该属性确实在当前运行时生效。若需重建场景，不把live属性写入读回当作实际pair table已更新。正对照必须覆盖实际选定形状，不能只靠其它方块。
5. **验收分类**：元数据false＋零事件仍为未决。只有精确形状/有效属性/时间覆盖与对照闭合后，才能审定是否属于“运行时本来禁用”；若运行时确实启用并产生连接处错误接触，则提交精确几何/配对修复方案审查，不实施整个腕部忽略。

当前用户上一轮明确“不提交GPU/PPO”，本轮“已授权”按恢复GitHub密钥处理，**不扩大解释成解除GPU限制**。本目录没有GPU提交入口，没有后台任务或自动重试。

## 实际测试与保持不变的状态

- `python -m unittest -v test_audit test_native_contact_observer`：28项CPU软件测试全部通过，包含原10项审计＋新增18项观测器测试。
- 原20个输入哈希再次通过；冻结997帧候选不变。旧7文件与W&B服务器回读哈希一致。
- 新运动求解0、GPU提交0、PPO0、过滤更改0、自动配对排除0。全局优化器仍是规格，未实现/未运行。
- 仍未得到物理穿透深度、实际配对因果或训练就绪证明。原4692严格相交和277未决拒绝状态不改变。