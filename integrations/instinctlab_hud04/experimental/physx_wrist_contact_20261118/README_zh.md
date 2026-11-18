并非已验收动作：HUD04 实际腕部碰撞过滤的隔离对照验证
==================================================

目标仍未完成。`accepted_reference=false; training_ready=false`。
此目录不是轨迹发布、训练入口或对历史任务的重试。

背景与唯一问题
--------------
上一步已证明左右腕各一对原始 nominal 碰撞体全角度恒相交；真实回收 USD 中对应关节 collisionEnabled 为未显式编写的 false。USD 值不等于实际 PhysX 过滤，不能自动忽略碰撞或验收参考动作。

本次 NEW、一次性 GPU 小测试，以当前最新父提交 dc04e1a 为基线：
发布前并行远端推进到3e2e1767，首次尝试在intent/POST/任务创建之前停下。已回读并保留全部10项变更：新增全形状端点不退化门禁明确拒绝005→008的460项分离丢失/348项新严格相交。此诊断不重做其求解/审查、不改变门禁；实际发布父提交应为3e2e1767，由发布回执绑定。
- 1 张 4090D、原已验证 IsaacSim5.1/IsaacLab2.3.2 镜像；3 个隔离 HUD04 副本、8 个显式 .005s 物理步，独立子进程最多240秒、日志8MiB、自动重试0。
- 使用原始固定头部29DOF模型/资产 SHA；不修改原始URDF/STL/动作/既有文件。
- 三副本重复相同诊断状态/关闭重力，只在两个腕roll关节的隔离USD覆盖层中分别保持原始fallback、显式true、显式false；不改变源模型。
- 使用官方 v2.3.2 contact_sensor.py 同款 SimulationManager physics view/create_rigid_contact_view/get_contact_force_matrix，精确body0/body1路径，绝不后缀猜测。
- 两个独立重叠自由box的实际接触力作为传感通路阳性对照。只有box阳性、该侧true阳性且原始/false均为零，才记录该侧特定腕body pair的有限过滤支持证据。零力本身不是过滤证明；force不是穿透深度；不是28形状对完整烹饪/过滤证明。
- 导入时contactreport阈值显式为0。保存全部8步的7通道数据、六精确关节绑定与前后属性、代码/资产身份。

退出与证据边界
--------------
历史366任务 status6/child -11 仍保持失败。本次是有区别的 NEW 实验：采用 AppLauncher 已支持的 native fast_shutdown=True，由父进程观察实际退出码并要求完整报告。没有os._exit/异常退出码改写。报告在关闭前落盘，以免丢失失败证据；父进程只有退出码0/无超时/完整严格绑定报告时才承认 isolated controlled_probe_completed。fast shutdown 正常退出不是所有扩展析构完成，graceful_full_cleanup_proven 永远false，不替代完整任务 clean-exit门槛。发生异常即使native close提前终止，也不会因缺报告而通过。

13项CPU测试仅为合成分类/契约测试，不是机器人测量或动作验收。官方接口源文件仅本地读取，发布清单记录其上游来源和hash，不将上游整个源文件打包。原有源码/历史候选/启动门禁均不改变。

不得因本探针成功而放行008或开始PPO。仍需真实非相邻/地形完整997帧可行性、原始导出与world/terrain单位对齐、校准支撑/材料帧、真实技能环境及动态验收。需要新任务时必须重新查询账户和项目全部非终态并排重。不得运行历史发布/创建/启动脚本；本目录启动器默认仅plan，新发布/创建脚本有独占intent，完成后不得重放。