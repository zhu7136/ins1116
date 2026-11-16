# HUD04 固定头部／胸口深度相机验证探针

这是 **独立资产验证程序**，不是已经训练完成的 HUD04 策略，也不替换本仓库已有训练入口。

## 退出问题的空场景对照

`lifecycle_probe.py` 默认仅打印计划，不初始化 GPU。显式执行
`python lifecycle_probe.py --execute-probe` 后，在同一个 GPU 任务中依次启动两个独立子进程：
仅 AppLauncher，以及不加载机器人或传感器的 SimulationContext（3 步）。
两者均调用正常的 `app.close(wait_for_replicator=False)`，分别记录返回和进程退出。
每个子进程初始化上限 90 秒、关闭上限 30 秒、总上限 120 秒，无自动重试。
截止后终止子进程只用于止损，**不算正常退出通过**；工作进程限时不是平台总费用硬上限。
CPU 检查：`python -m unittest -v test_cpu.py test_lifecycle_cpu.py`。
诊断不加载 HUD04、不生成深度测量，不进行策略训练。

## 范围

- HUD04 串联模型，29个可动身体关节；头部 yaw/pitch 在模型零位改为 fixed，不存在可供策略控制的头部动作自由度。
- 固定根基座仅用于此诊断探针，不能作为自由站立、平衡或接触稳定性的证据。
- 原始胸口深度使用标准 IsaacLab RayCasterCamera，对独立静态平面采样，形状 `[1,27,48,1]`。
- 检查关节集合、头部导入零位与相对变换、有限值、传感器更新及目标有效像素。
- 不调用 Instinct-RL、参考动作、PPO、训练任务注册或外部下载。
- 不验证 InstinctLab 自定义 grouped/noisy 深度管线、18×32策略输入、奖励、重置或技能成功率。

## 离线检查（不会初始化 Isaac/CUDA）

在本目录执行：

```sh
python -m unittest -v test_cpu.py
python preflight.py
python worker.py
```

发布前26项CPU测试通过，包含纯CPU伪子进程的正常退出及关闭超时回传测试；这不是GPU运行结果。`manifest.json`记录实际发布输入的SHA256。

## 有界GPU探针

需要预装兼容的IsaacSim/IsaacLab GPU环境。候选接口对照为IsaacLab v2.3.2，源码依据见`SOURCE_MANIFEST.json`与`API_COMPATIBILITY_REVIEW.md`。

```sh
python worker.py --execute-probe
```

若当前Python不能导入IsaacLab，应使用环境实际的Python启动器：

```sh
python worker.py --execute-probe --isaac-launcher /isaac-sim/python.sh
```

启动器路径是候选，不保证所有镜像相同。无启动器即失败，不在线安装或升级Torch。
最多120个0.005秒物理步；初始化300秒、子进程540秒超时停止，不自动重试。
仿真器关闭阶段另设30秒上限，不延长540秒总上限。本RayCaster探针没有Replicator写出作业，关闭使用`app.close(wait_for_replicator=False)`，不跳过正常清理。该参数已对照IsaacSim v5.1.0官方源码（commit `47d886f2858d1ceed556b21c88927aa67bc81c12`，`source/extensions/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py`）；这不代表关闭卡住的根因已确定或修复已通过GPU实测。
本探针仅使用Warp/CUDA射线深度，不使用RTX图像相机：显式设置`ENABLE_CAMERAS=0`、`enable_cameras=False`并用`sim.step(render=False)`推进物理；相机仍通过`camera.update`更新射线。此改动用于单独验证关闭问题是否与不需要的渲染流程有关，不是对根因的既定结论。报告显式记录CUDA设备与RTX开关。
worker看门狗不是平台费用或clone/启动/释放时长的硬上限。
本仓库的Flux入口为`gm-run ins1116/flux/hud04_asset_camera_probe/worker.py --execute-probe`。完整平台日志已确认克隆目录名为`ins1116`，不能添加GitHub所有者目录。

## 结果

成功后保存`output/report.json`、`output/depth_samples.npz`及`worker_result.json`；原始子进程日志保留在worker本地。worker每30秒输出一次进度，失败时将脱敏后的最多8192字符子进程日志尾部输出到平台日志。
只有正常退出且验收通过才报告worker的`probe_passed=true`及`probe_lifecycle_passed=true`；`training_ready`始终为false。`simulation_checks_passed`单独记录物理/相机检查结果，不等于整个任务成功。
三个结果文件还会作为带整体与逐文件SHA256的JSON分块输出到应用日志；原始文件总量上限128KiB，不导出其他文件。
只要仿真检查已完成，即使后续关闭超时，也回传这三个文件，其中`worker_result.json`明确保留失败状态；不会因为存在仿真报告就把关闭超时改判成功。

获取完整平台日志后，可离线恢复：

```sh
python evidence_transport.py application.log --output recovered_probe
```

缺块、重复、大小或校验不符将拒收；支持SDK前缀与拼接JSON，解析日志上限10MiB。CPU已验证传输逻辑，实际平台完整日志传输仍须验证。

## 来源、许可与修改说明

- InstinctLab：https://github.com/project-instinct/InstinctLab ，基线`de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057`。许可原文保存在`licenses/InstinctLab-LICENSE`，CC BY-NC 4.0，保留署名及非商业条件。
- HUD04资产：https://github.com/limxdynamics/humanoid-description ，基线`4c0a3c71200b810a52831c7c2b2de0ed6ce819a4`。许可原文保存在`licenses/humanoid-description-LICENSE`，Apache 2.0。
- 控制参数是参考公开LimX GMR配置的初始值，不是调优结果或硬件安全参数。
- 本分支新增探针、固定头部URDF和身体碰撞近似，修改日期2026-11-16；这些修改不能仅由上游基线commit代表，以文件manifest为准。
- 本目录不含账号凭据、签名下载链接、账单信息、私有数据、正式动作数据或训练checkpoint。