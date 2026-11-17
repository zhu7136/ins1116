# HUD04：真实导入碰撞过滤的只读证据采集器

本轮基线：`39650cbae894aacacc6f7d896e31dcc70ca5d417`。本轮没有重复几何距离探针，没有执行旧创建/发布/启动脚本，也没有提交GPU任务。

## 新发现与实现

1. **全局 self-collision 开启不等于所有相邻刚体都碰撞。** OpenUSD Joint 的 `physics:collisionEnabled` 默认false，而 CollisionAPI 的同名属性默认true。两者不能混用；还要检查关节是否启用、FilteredPairsAPI、碰撞组与运行时环境隔离。固定版本OpenUSD v25.05 schema原文已Gitblob/SHA256校验；本轮CPU安装usd-core25.11并真实调用其C++schema，确认这两个fallback。该CPU库/参考版本不是历史IsaacSim镜像的USD/PhysX构建证明。
2. **精确列明需要观测的28个相邻形状对，覆盖6对直接关节连接的碰撞链接。** 从原903对清单和原URDF按父子链接精确绑定到关节名，模型/配置/清单/原重叠见证哈希都验证。不从图关系豁免，全部903记录仍保留；原两项腕部圆柱—手凸包代数姿态重叠仍保留，不宣判真实物理碰撞违规，也不把它们强制当成可碰撞分离约束。
3. **新增 `usd_filter_snapshot.py`：只读观察已有stage，不导入机器人、不启动仿真、不创建/修改碰撞属性。** 用真实USD API遍历组合后的实例代理，记录每个关节body0/body1、collisionEnabled/jointEnabled的解析值与是否显式写入，记录根self-collision属性、碰撞几何类型/参数/局部mesh点哈希、刚体归属、世界变换、接触偏移、全局filteredPairs与碰撞组集合。世界矩阵明确保留Gf行向量约定，不悄悄按列向量使用。
4. USD mesh点/convexHull近似声明不是PhysX cooked顶点；stage的metersPerUnit不是源NPZ/STL单位/对齐证明。过滤元数据采集不等于运行时GPU环境ID、实际pair过滤或接触测试。输出始终 `actual_physx_pair_filters_verified=false` / `cooked_geometry_verified=false`，不会写accepted动作、轨迹或清单。

## 本轮实际执行

- **26项CPU测试通过**，0.036秒。不是旧测试数累加，也不是26项物理验证。覆盖真实USD默认/显式值差异、缺失PhysX schema、外部过滤owner、碰撞组集合、instance proxy、同名刚体歧义、静态碰撞体、源单位不推断、Gf变换、禁用关节与碰撞体、非有限尺寸拒绝、prim/mesh点/拓扑预算、输出禁止覆盖、验收状态不得晋升。
- 采集上限：默认4096个prim、累计65536个mesh点、累计12×点预算的拓扑项，JSON最多512000字节；超限报错，不把部分结果标成完整。真实大型场景需要明确缩小范围或审查预算，不能自动无限扩展。此为可信输入的CPU只读工具，不是恶意USD解析沙箱或GPU外层账单限制。
- **Flux实时只读查询**：2026-11-17 12:28:53 UTC，账户状态1–4、目标项目状态0–4两组查询均total0/hasNextfalse；不是永久提交许可，不等于资源容量或费用上限。
- **完整读取已完成 TASK_20261117_016 日志**：124484字节，SHA256 `2690a7443f62f5d216d2061ed62ba13466163f8ae6f8f142aa3e9b75fe30c880`。仅从新鲜task-info确认的HTTPS签名对象地址下载，独立无账号认证session，禁止跳转，4MB/30秒预算；签名URL及原始日志未落盘，留下安全摘要与日志哈希。历史任务仍status5，没有重跑。日志未恢复出实际Python启动器/cooked几何/过滤信息，不把历史控制短测晋升为这些验证。
- 固定IsaacLab commit `f73c33173801f5f8afea4142482e47b7710c2b75` 的schemas/from_files/interactive_scene源码只读下载并校验，不执行。该源码明确GPU env-id过滤与USD碰撞组过滤路径不同，实际场景仍需观测。

## 使用边界与下一步

独立工具依赖 `usd-core==25.11` 供CPU测试；在真实IsaacSim进程内，应使用该进程自带pxr。不要把CPU环境的USD库安装进历史镜像去改变其行为。

在真实、已完成导入的HUD04 stage上调用：

```python
from usd_filter_snapshot import collect, write_new
# stage和exact_robot_root必须取自真实场景，不在此工具内伪造或猜测。
snapshot = collect(stage, exact_robot_root)
write_new(new_report_path, snapshot)
```

在实际导入后和clone/filter完成后各采一次；完整命名、真实root和依赖哈希须由新运行绑定。`compare_inventory`仅检查精确刚体叶名/数量并拒绝同名歧义，不用碰撞shape名称后缀猜索引，数量一致也不证明cooked形状一致。任意USD缺失记录、组策略或GPU过滤仍记UNRESOLVED，禁止从“没有显式filter”推出必然碰撞。

本轮 `SYNTHETIC_USD_SCHEMA_OBSERVATION.json` 明确是合成CPU fixture，**不是HUD04导入结果**；`FILTER_EVIDENCE_REVIEW_20261117.json`是哈希绑定的28对观测契约。尚无真实导入USD；尚未证明平台硬停止/预算及实际launcher，因此未创建新GPU任务。未来提交前需再查实时任务/排重并确认这些执行边界，绝不重提TASK_20261116_732/TASK_20261117_016。

保留29个身体动作、头部固定零位且0动作、原数据、90Hz、模型限值。源导出/root/四元数/世界单位和可审计支撑区间门禁仍保留；不按小间隙选择坐标假设。GPU create/start/stop/retry=0，重定向solver_calls=0，PPO_updates=0，`accepted_reference=false`，`training_ready=false`。W&B未重试或写入，不假定invalid audience已恢复。