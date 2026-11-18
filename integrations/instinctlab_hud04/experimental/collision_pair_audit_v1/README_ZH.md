# 两对腕部碰撞审计 v1：证据整理完成，真实配对因果仍未闭合

本轮基于 `fcf38047fce7eba1ea893fc756334f3bdc0a097a`，遵循用户决定：不再增加旧LP轮数、不提交GPU/PPO；先审计实际碰撞配对，再重构全轨迹约束。没有把用户的分析解释成修改碰撞过滤或模型的授权。

## 冻结对象

`material_trajectory_v1/TRIAL_MATERIAL_003_FULL_SEQUENCE/FULL997_ENGINEERING_CANDIDATE_UNACCEPTED.npz`，SHA256 `2b4070e32e3a5f120caea7942b42e91a6f31bbca9e0b6d7e8e135b50803ad2f4`。

逐字节复核未变，仍不合格：4692条严格相交，277条未决，未生成/替换训练参考。计数是帧—形状对／地形三角身份，不是接触帧数、力或穿透深度。

## 精确配对与形状

| 左／右 | 形状一（从0计数） | 形状二（从0计数） | 连接关节 |
|---|---|---|---|
| 左 | left_wrist_pitch_link#1 | left_wrist_roll_link#0 | left_wrist_roll_joint |
| 右 | right_wrist_pitch_link#1 | right_wrist_roll_link#0 | right_wrist_roll_joint |

每侧形状一的XML名称为 `official_{side}_wrist_pitch_link_2`，类型为圆柱，半径0.029、长度0.06，几何原点xyz=(0,0,-0.065)、rpy=(0,1.57,0)。形状二为 `official_{side}_wrist_roll_link_1`，原始对应STL，scale=(1,1,1)，局部原点为单位变换。名义计算使用该STL凸包；历史USD声明convexHull，但cooked顶点并未证实相同。

视觉模型不是该圆柱：两个链接的visual均为各自原始STL。AUDIT.json保存全部四个visual文件、两手collision网格的仓库路径/哈希、XML几何和变换，不用visual取代collision。

两侧零位 parent_T_child 的旋转为I，平移(0,0,-0.065)。roll轴为X，任意角度变换为 T_zero·Rx(q)，**完整相对变换并非固定**；不变量是公共内点位于轴上。独立从原始二进制STL重建凸包、重算公共内点到圆柱/凸包平面内侧余量均为正。该连续论证只证明上述名义形状无法通过关节运动分离，不证明实际PhysX接触。

## 运行时审计结论

1. SHA绑定任务366的三阶段真实USD快照。按精确关节路径、body0/body1、XML collision名称及刚体归属绑定；两侧关节collisionEnabled均为fallback false，单独碰撞形状属性为true。**关节级与形状级开关不是同一语义。** 保存原始列向量转换后的相对几何矩阵；Gf原始行向量记录也保留。任务366本身失败/exit -11，不标成成功运行。
2. 任务138历史实测：每侧原articulation body-pair为8步0点/0力，独立副本每步3点并有非零力，阳性方块对照有接触。当前只重读已保存报告，不重跑任务。
3. 仍缺精确shape-ID/cooked等价、reset后其它形状禁用读回，以及无其它上下文变量变化的因果对照。**结论仍是“元数据关闭意图＋上下文差异已观察到”，不能将用户所列三种情况中的第一种标为已证实。** 更不能据此授予两对之外的豁免。
4. **穿透量：未知／未测得。** 旧报告没有可靠有符号间距或形状级穿透深度。公共内点余量、接触点数和接触力都不是穿透量；不把约0.0287的内部几何余量写成28.7mm物理穿透。

## 16处窗口连接诊断

BOUNDARY_DIAGNOSTICS.json检查全部16处窗口起始连接附近的差分。最明显的示例：frame351处前后关节差分速度最大差27.0271 rad/s；frame292处左右二阶差分最大差3959.6237 rad/s²。两者是离散变化，不是连续插值跳变的证明，也不能把变化全部归因于窗口优化。

全局关节二阶差分峰值3415.7206 rad/s²，根平移二阶差分范数峰值684.5364 m/s²；与既有审查一致。997个共享保存状态不代表C2插值已通过，当前没有已绑定的物理加速度验收限值。

## 下一阶段规格，不冒称实现

`GLOBAL_TRANSCRIPTION_DESIGN_ZH.md`已写明：一个共享全局轨迹/力问题、所有接口位置速度加速度一致、根SE(3)处理、非穿透/支撑/力矩/浮动基座动力学硬约束、phase-I仅诊断、信赖域SCP与真实非线性复核、段内扫掠验收。

**该文件是重构设计，不是已实现或运行的新优化器。** 当前24个提议支撑窗口与346个未标注帧仍不能直接当作真实接触模式。

运行时审计下一步需在同源原articulation中获取实际shape-ID、cooked几何、带符号距离/接触分离及最终关节过滤状态，并核对原碰撞属性和其它形状开关。若API可观察能力不足，保持UNRESOLVED；不拿body-pair零接触替代。必要的开关因果对照须保持精确原形状/惯性/根/关节状态，仅改变指定关节碰撞开关，并保留同形状阳性对照、全部签名和反例。能力、授权和有限预算审定前不提交任何GPU任务。若确认运行时启用且出现错误连接接触，再审定仅针对精确几何或这两个配对的最小修复；本轮不实施修复。

## 实际执行

只有CPU只读审计、凸包数值复核和差分诊断。20个输入文件哈希记录并回读；10项CPU测试通过。新运动优化器调用0、GPU任务0、PPO0、自动排除0。未重放任何旧发布/创建/启动/求解intent；未查询新的Flux全账户状态，因此不宣称账户当前无其它活动任务。

`python -m unittest -v test_audit` 可重跑只读测试；`audit.py`的输出禁止覆盖，不能直接重跑。输出JSON与日志随代码保存。accepted_reference=false；training_ready=false。

## 2026-11-18 14:14续：直接接触点材质身份路径

### 新确认的接口路径

对既有NVIDIA107.3官方physx.html快照重新校验SHA256 e01be9c0…，提取8个精确属性条目。ContactData.material0/material1明确为uint64，使用PhysicsSchemaTools.intToSdfPath解码；同一接触头提供collider0/1及contact_data_offset/num_contact_data。

因此可准备另一条**不依赖材质张量槽排序的接触点身份采集路径**：

`同一事件的actor/collider0或1 → 该头指向的接触点 → 同侧material0或1原生ID → 解码材质prim路径`

这条路径尚未获得目标运行时样本，也不建立任何张量槽到shape映射。解码材质路径本身不证明材质数值、组合模式、最终求解器值或有效摩擦。

### 实现与回归

- 新增contact_material_links_1073.py，现有NativeContactObserver在原回调内复制每个点的材质ID、解码状态、face_index0/1。沿用已有stage/proto校验及事件/点预算，不另订阅或创建进程，不步进、设置材质、改变报告阈值或碰撞规则。
- 保留原事件side0/side1顺序；反向匹配registry时不偷偷对调材质。相同collider的不同接触点可以保留不同材质ID和面索引，不按body或collider合并材质。
- 材质路径可在/Looks、/physicsScene等命名空间，不强制/World；解析器只是保守绝对prim路径子集，非USD材质绑定解析器。
- 缺失字段、0ID、无效ID、解码失败或不支持路径保留UNRESOLVED状态，不推断默认摩擦或把接触删掉。0ID含义未获接口条目确认，不擅自设为默认材质。面索引负值/全1值保留，不解释sentinel或原始STL/cooked三角形对应关系。
- 12项新增软件测试；本目录全部97项通过(.173s)，含旧订阅/预算/阶段身份等回归。只用假回调对象，没有实际运行时采集。97不是物理测试数量；与其他目录50项软件回归分开记账。

证据：CONTACT_MATERIAL_LINK_API_EVIDENCE.json、TEST_CONTACT_MATERIAL_EXECUTION.log。收集脚本collect_contact_material_evidence.py已经执行，独占输出，不能盲目重放。输出记录4个代码哈希，读回全部一致。

### 旧准备记录的版本边界

native_contact_observer.py发生4行改动，旧RUNTIME_PREPARATION_MANIFEST.json对该文件的哈希已不匹配，其余3项匹配。旧清单保留为历史记录，**不得拿它作为当前采集器版本的运行准备/授权证明**，本轮不覆盖或伪造新运行准备回执。新辅助模块也须进入未来完整部署清单。

所有本轮修改仍本地未提交/发布。新GPU任务0、原生接触材质样本0、优化器0、新动作0、PPO0；未重新查询Flux当前任务状态，不能宣称账户无其他任务。accepted_reference=false、training_ready=false，有效配对摩擦仍null。

后续任何GPU观测仍以“已配置且可核验的服务端300秒强制终止”为启动前置条件；客户端超时不计入该证据。目前尚未满足此前置条件。必要短GPU授权不重复索取。本阶段仅准备采集能力，不构成已运行、持续后台运行或可接受参考动作。