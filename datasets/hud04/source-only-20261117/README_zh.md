# 原始动作与无损整理归档 — SOURCE ONLY / NOT TRAINING READY

这不是 HUD04 重定向参考动作，不可导入训练。
保存四段原始 G1 动作（共3428帧、90Hz）、配套原始 STL/YAML、无损按名称排列的派生数据、溯源与审查证据。
original/ 7个文件逐字节保持，source_staging/ 可逆但不是标准参考动作字段。
没有关节角裁剪、重采样、符号变换、根坐标变换或四元数重排。

Step1 997帧仍有261帧超出HUD04位置限值、165个超速区间，尚无验收通过的HUD04动作。
官方发布消费接口有torso/wxyz绑定；原始exporter执行、原始世界单位/对齐未完整恢复。
原metadata.yaml引用缺失的Step6；不得把它当作完整可用的数据集元数据。
metadata_kneelClimbStep1.yaml与配对Step1/STL链接已恢复。
历史报告保留原时点判断，不能把不同报告时间的进度混为最终验收。

accepted_reference=false; training_ready=false; retarget_solver_calls=0; ppo_updates=0。
头部yaw/pitch固定零，不进入动作；29身体动作与模型限值保持不变。
后续：完成目标几何/接触策略与有界重定向，再独立检查全序列及真实匹配地形回放。
