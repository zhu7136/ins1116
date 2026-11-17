# 源动作下载核对与无损转换准备（不是 HUD04 参考动作）

基线：`bf4a057178061f040e124ab6d48d55954ad5504f`。本次只新增工具、测试和记录，
不修改资产、训练配置、动作验收门禁或此前几何报告。

## 实际完成

重新读取原始公开 Drive 文件夹，列出的仍是既有七个文件：四段 NPZ、两份 YAML、
一份 STL；未发现 PKL、导出脚本、生成命令或坐标约定附件。七个已下载原件的 SHA256
均与 `MOTION_REVIEW_20261117.json` 匹配，故不重复下载/覆盖这些原件。

来源：<https://drive.google.com/drive/folders/11FgL1RjM32bNNR1d1M1_Wgfntz7Hxtst>

新增下载并按 Git blob 验证 InstinctLab 的 `scripts/GMR_to_instinct.py`、说明文档和
Project Instinct 两种编辑器的输入输出实现；它们是来源候选证据，不是 Step1 的导出证明。

|源动作|帧数|原生 Hz|与 HUD04 位置限值冲突帧|与 HUD04 速度限值冲突区间|
|---|---:|---:|---:|---:|
|jumpsit2-ziwen-retargeted.npz|967|90|484|106|
|kneelClimbStep1-x-0.1-ziwen-retargeted.npz|997|90|261|165|
|rollVault11-ziwen-retargeted.npz|942|90|585|131|
|superheroLanding-retargeted.npz|522|90|115|100|

四段合计 **3,428 帧**。每段生成单独 `.source-stage.npz`：

- 29 身体通道按名称无损排列，并保存正反列置换；**这不是跨机器人几何重定向**。
- 保留原关节值、浮点精度、原始根位置/四元数、framerate；生成 `frame_index/90` 时间轴。
- 额外生成与 HUD04 模型限值比较的布尔掩码，保留全部超限，不做裁剪、平滑或重采样。
- 写盘后实际回读每个数组并检查 dtype 和逐元素相等，逆置换恢复源关节数组。
- 输出中故意不提供标准参考动作字段 `joint_pos/base_pos_w/base_quat_w/framerate`，
  不采用 `retargeted.npz` 结尾，避免当作可训练 HUD04 动作误载。
- head_yaw/head_pitch 锁零只记录为未来目标约束，不加入 29 维动作，也不修改源四元数。
- 无 `accepted` 清单，无求解、GPU 或 PPO。

`SOURCE_STAGING_REPORT_20261117.json` 是读回验证后的清单，不是数据负载或验收证明。
本次不将原始/派生动作数据重新分发到 GitHub；转换准备脚本和报告可公开复现步骤。

## 新恢复的来源证据与边界

1. InstinctLab `de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057` 的
   [`GMR_to_instinct.py`](https://github.com/project-instinct/InstinctLab/blob/de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057/scripts/GMR_to_instinct.py)
   将 PKL `root_rot` 从 xyzw 改为 wxyz；默认 pelvis→torso_link，使用逐帧 FK，
   默认反转 waist_yaw/waist_roll/waist_pitch。**不执行此脚本，也不将这些规则再次施加给现有 NPZ**。
   未恢复到 Step1 的原始 PKL 或命令，不能断言它使用了该导出器及默认参数。
2. `robot-motion-editor` 的
   [`bee9fc67773b80fb5daafed5726498c45e3d0bdc`](https://github.com/project-instinct/robot-motion-editor/tree/bee9fc67773b80fb5daafed5726498c45e3d0bdc)
   文档/Three.js 显示约定是 wxyz，并允许用户指定 URDF；后端保存 float64。
   `robot_trajectory_editor` 的
   [`2c7bba9b6a76c0a1afcf99bb5d99a3944f3946ca`](https://github.com/project-instinct/robot_trajectory_editor/tree/2c7bba9b6a76c0a1afcf99bb5d99a3944f3946ca)
   当前后端保存 float32。不同文件精度可以兼容不同处理路径，**不能靠 dtype 推断导出器或根坐标**。
3. InstinctLab README 指定的 IsaacLab 提交实际是
   `f73c33173801f5f8afea4142482e47b7710c2b75`，不是仅有版本徽章的 v2.3.2。
   新下载该提交的 `terrain_generator.py` 并校验，SHA256 为
   `a364d85812a7713b21a94a450f40e8ef5d448bbe652bcb2ee36184b6300c409b`，
   与此前审查的 v2.3.2 **该文件逐字节相同**。不代表两个完整版本等同，
   也不代表恢复了旧 GPU 环境的安装证明。
4. 检查了 `omniretargeting` 的锁定提交
   `6518c36a2be9dab2538cc1b95840da00fbd6d4b7` 文档；未安装或执行其求解器，
   不启用默认稳定化、场景缩放或其他可能改变原始数据的路径。

## 执行与测试

在交付仓库根目录，设置原始数据目录及一个不存在的新输出目录后：

```bash
export HUD04_DELIVERY_ROOT="$PWD"
python integrations/instinctlab_hud04/prepare_source_conversion.py \
  --dataset "$HUD04_SOURCE_DATASET" \
  --urdf flux/hud04_asset_camera_probe/humanoid-description/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf \
  --output "$HUD04_STAGING_OUTPUT"
```

输入或模型校验失败则拒绝继续；输出目录存在则拒绝覆盖。依赖 NumPy、PyYAML 及既有
交付工具。完整 CPU 测试还需要 trimesh/shapely。

本次实际沙箱输出 `/workspace/hud04-source-staging-20261117`，用户不能直接访问该路径；
故在聊天提供结果并发布可复现脚本和清单，不把仅本地文件描述为可下载交付物。

新增 **12 项测试通过**。首次完整运行 162 项时 7 项因缺少 trimesh 报错；
补装与上轮一致的 trimesh 5.2.1、shapely 2.2.0 后，重新执行 **162 项全部通过**。
这些是 CPU 软件/格式检查，不是运动学重定向或物理验收。

## 不能冒充已经完成的部分

`accepted_reference=false`，`training_ready=false`。现有四组根/四元数假设仍未选定。
还需要绑定 Step1 原始导出命令/版本、生成模型、根坐标、四元数顺序、世界与 STL 单位/对齐，
并审查手足膝材料点的支撑约束，才能进入有界重定向。
上游通用约定、分布看起来合理、几何距离较好均不能替代这份绑定证据。