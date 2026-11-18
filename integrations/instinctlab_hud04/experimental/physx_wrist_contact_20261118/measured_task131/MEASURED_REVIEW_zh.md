# 腕接触 A/B/C 实测回收：阳性对照失败，不得据此豁免碰撞

## 已核实
- NEW TASK_20261118_131，实际源码5ec4d476986b2629c2df18cd6fe5d316c3edf022，父提交3e2e1767ff3c60699bf7801a5c84cbe06dc605e9的并行全身不退化门禁完整保留。第一次发布尝试在任何intent/POST/任务创建前因远端推进停止，随后回读10项并行变更再发布；没有覆盖新代码。
- 1张4090D24GB，IsaacSim5.1/IsaacLab2.3.2；3隔离副本，8个显式0.005s步；独立子进程上限240秒、日志8MiB、自动重试0。没有新增任务重试。
- 平台终态5再次读回，CST10:20:37开始/10:22:42结束（125秒为平台时间跨度，不是账单证明）；子进程17.416424851864576秒，return0、无超时、直接子进程已回收。使用官方native fast_shutdown=True；不宣称完整析构/graceful full cleanup。历史366/-11失败不被改写。
- 原始模型、资产配置hash及6个精确关节/body0/body1绑定已校验。29身体DOF、头动作0；七通道各8样本。
- 七通道全部为0，包括独立方块阳性对照；重新解码报告逐样本复核。**结论INCONCLUSIVE_POSITIVE_CONTROL_FAILED**。controlled_probe_completed只表示受控采集程序完成，不是对照有效或过滤验证通过；两侧filter_supported均false。不能把零力当接触过滤证明，不能白名单腕部，不能放行008或参考动作/PPO。
- 完整平台日志156987字节、SHA256 3915dd8638c4bd56348e383ab9f68f4100450dce01a9757aa5dcfcf590a15839；签名URL和原始全日志未保存/发布。contact_report、supervisor_result通过字节SHA绑定回收；安全审查只含脱敏尾部。子日志hash见supervisor_result。

## 新找到的仪器差异与下一次修复，不是已证根因
官方v2.3.2 simulation_context.py:694–705默认将`/physics/disableContactProcessing`置True；contact_sensor.py:75–79构造时置False。任务131仅借用raw physics tensor view，没有构造ContactSensor，也没执行该设置。官方注释同时说此开关可能只影响CPU、GPU通常始终处理接触，因此不能断言它解释了GPU全零。

另一个采样风险：两box只在reset前摆放一次，没有在每个采样步重置或记录其实际位姿/速度；reset可能先行解穿透。下次独立修订需要：
1. before reset启用接触处理并回读；优先与真实ContactSensor通路交叉检查。
2. 对每步box重叠状态做显式重置和位姿/速度读回；官方RigidObject v2.3.2 write_root_pose_to_sim使用root_physx_view.set_transforms(poses_xyzw,indices=...)，速度使用set_velocities。不能把wxyz直接给native transform。
3. 阈值0、精确过滤目标和完整通道保持；记录步前后真实重叠、力矩阵、详细contact计数/溢出警告；只有box阳性及true组阳性才能解释原始/false零力。
4. 原始模型/动作/限值不改，所有仪器覆盖仅隔离USD层。当前仅新增contact_processing_guard.py修复原语、6项CPU测试；**尚未接入、执行或验证任何新GPU修复**。本次19项CPU合成/契约测试PASS，不是19项物理验收。

## 下一阶段硬约束仍然成立
008被并行全形状门禁拒绝（005→008，460项已证分离丢失/348项新严格相交）；腕部仪器不推翻它，不重做并行求解或擅改候选。

保持997帧90Hz原始数据、29身体动作、头固定零、模型限位/速度、不裁剪、不伪造acceptance。当前仍无已验收HUD04参考。下一步先修独立接触阳性对照，再结合完整地形/全身不退化约束做有界重定向；导出/单位/根/四元数和材料支撑证据未解决的部分不得由物理零力替代。PPO更新0，accepted_reference=false，training_ready=false。本轮未尝试W&B鉴权，历史invalid audience不假定已恢复。

任何后续GPU创建前必须重新实时查询排重；禁止重放已存在intent的发布/创建/启动脚本和历史任务。最后空闲查询的JSON随证据回执保存，不能作为未来提交的实时证据。