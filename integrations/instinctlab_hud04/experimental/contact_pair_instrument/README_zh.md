# HUD04 接触对独立仪器：未物理验证候选

本目录为新的隔离候选，不替换既有资产/控制探针、任务注册或参考动作门禁；**不要重提TASK_20261116_732/TASK_20261117_016**。

## 做了什么

- `contact_instrument.py`：不加载参考动作的双环境/四刚体/静态三角Mesh仪器。下层1kg，上层2kg；检查地形/其他身体法向力与净力，抬起/恢复检查接触消失及缓存。260物理步、dt=.005。默认只打印计划，不导入Isaac或初始化CUDA。
- `terrain_contact_bridge.py`：从更大的本地隔离Terrain候选提取的接触数组归约/严格USD census/endpoint规划代码。仪器仅调用`_inventory`、`make_pair_plan`、`reduce_contact_points`；其中完整任务安装器依赖尚未集成的Terrain候选，**不能从本目录调用它来启动训练**。
- `contact_instrument_worker.py`：独立新worker，不运行旧脚本。显式固定提交、HEAD内manifest及输入哈希、显式已核实launcher、独占输出；初始化180秒/子进程300秒/退出20秒，TERM/KILL各3秒；零重试。worker只管理自己启动的子进程，不是Flux API客户端或平台账单上限。
- `CONTACT_INSTRUMENT_MANIFEST.json`：三项执行输入SHA256。`test_contact_instrument.py`：29项CPU软件检查；默认计划入口也被实际执行，未初始化GPU。

CPU检查（仓库中进入本目录）：`python -B -m unittest -v test_contact_instrument.py`。

默认只打印计划：`python -B contact_instrument.py`、`python -B contact_instrument_worker.py`。

这里故意不提供已可直接提交的GPU命令：尚未恢复Flux实时任务查询协议、核实镜像/launcher和平台超时取消路径。提交前必须完成实时排重；secret已授权不代表API入口已恢复。任何显式执行还会要求字面40位预期提交和新输出目录。

## 不能据此宣称通过的事项

尚未执行IsaacLab/PhysX/GPU；未知镜像兼容性、静态Mesh过滤、全局GPU缓冲区与清理路径真实表现。已知质量15%/0.25N诊断容差不是HUD04接触/支撑/安全阈值。

空间分离的自由刚体不是实际环境碰撞组隔离或机器人自碰撞；未验证HUD04的19碰撞链接/43形状、固定头导入、实际任务cooking、scene子步collector。未加载、生成、验收任何参考动作，没有重定向/PPO/W&B新运行。

更大的本地Terrain接触候选与这29项一起曾实际执行121项CPU测试；本目录单独可重跑的是29项，不能把其余未打包的92项说成本目录交付测试。完整测试计数不代表物理有效性。

下一条训练关键路径仍是：真实来源/单位/对齐与接触许可 -> 有界HUD04参考生产与独立碰撞审查 -> 匹配地形短回放/感知与并行环境验证 -> 2环境x24steps、1次PPO更新的链路smoke。没有前项证据不得塞synthetic fixture或原G1动作来通过门禁。

`accepted_reference=false; training_ready=false; solver_calls=0; GPU_submissions=0; PPO_updates=0`。