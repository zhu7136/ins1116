# InstinctLab 的 HUD04 专用集成（第一版候选）

这不是再跑一次独立资产探针。这里将已验证的 HUD04 资产接到原 InstinctLab
的感知运动跟踪环境，新增 HUD04 Plane / PLAY 配置，**不修改 G1**。

## 2026-11-17 后续更新：原始数据恢复、实际裁剪与三维表面核验

已恢复原始 4 个 NPZ、2 个 YAML、1 个 STL，7 项 SHA256 与既有记录一致，
计算前后数据不变。旧 6696 点报告重算后的 SHA256 与历史记录**逐字节一致**。
新增 **26 项** CPU 回归测试，现合计 **150 项通过**；没有运行 GPU、重定向或 PPO。

按显式条件尺寸 `(9,12)` 重建地形：8087 面裁剪为 8068 面，边缘 62 个顶点
z 均值为 **0.135141808 原始单位**，边缘并不平坦。真实上游函数 AST 回放
与镜像处理的 faces、移动后 vertices、origin 均完全一致；但这不是历史 GPU 环境复测。
这 6696 个点的裁剪前后上包络没有变化，共同平移的间隙误差不超过 `2.23e-16`。

另完成名义源 G1 胶囊轴段到三角面的三维表面距离及侧向面查询，输出逐帧诊断和
**几何候选区间，不是支撑/接触标签**。24 个最小间隙见证、48 项独立距离复核通过。
详细输入、阈值、版本、数值和局限见 `PROCESSED_TERRAIN_REVIEW_20261117.md`，
完整结果及独立复核见同名前缀的 `SURFACES` / `INDEPENDENT_CHECK` JSON。

**导出脚本/根坐标/四元数顺序/世界单位和对齐证据仍未恢复**；NPZ 不含这些字段。
四种坐标假设未选定；垂向间隙和无符号表面距离都不能用于自动验收动作。
下一步先取得同源导出/对齐证据，再审查真实支撑约束和进入有界重定向。
`accepted_reference=false`、`training_ready=false`。以下 124 项测试和“原始数据未恢复”
等描述是上一个恢复阶段的历史记录，不代表本节后的最新恢复状态。

## 2026-11-17 恢复续接（上一阶段记录）：保持原始参考频率，补核地形变换链

从远端 `fd12fe13ce9293903d04e83d974987025f03b75e` 恢复全部 85 个仓库文件；
此前本地工作树、原始动作/地形数据和详细逐点报告不在本轮环境内。
已重新运行原有 **123 项 CPU 测试通过**，但尚未重新计算真实数据的 6696 个探测点。
此前测试数与数据复核结果不能混称为本轮重新测得的证据。

发现并修正一项频率契约问题：原 overlay 开启 `motion_interpolate_bilinear`，
却继承 `AmassMotionCfg.motion_target_framerate=50.0`，会在加载后对参考重采样。
现显式设置 `motion_interpolate_func=None, skip_frames=0`，保留文件原始帧数和频率，
不改变 50 Hz 控制/200 Hz 物理设置，不改源 NPZ，不放宽位置/速度门禁。
新增一项执行真实 `_motion_cfg` 函数体的 CPU stub 回归测试，合计 **124 项通过**。
此测试不等于真实 IsaacLab 导入、参考动作验收或动力学验证。

地形裁剪、边缘均值、生成器布局平移、动作原点及墙面证据见
`TERRAIN_TRANSFORM_RECOVERY_REVIEW_20261117.md`。这是源码契约审查，
不是已完成真实 STL 裁剪/边缘高度计算或接触回放。
未恢复原始数据前，不生成支撑区间、候选动作或坐标选择。
本轮无 GPU 提交、无求解、无 PPO，`training_ready=false`。

## 状态与边界

- 上游仓库：<https://github.com/project-instinct/InstinctLab>，固定提交
  `de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057`；候选环境为 IsaacLab v2.3.2。
- 已通过 36 项 CPU 软件测试；已在该提交的真实、稀疏 Git 检出中安装并逐文件回读，
  原有受跟踪源码无变化。稀疏检出仅用于安装检查，不是可运行的完整仿真环境。
- 已验证探针对应源 `8b421f55`，这里按 SHA256 复用其资产、控制契约和相机模块，
  不重写 PD、不改旧探针、不运行旧创建/发布/启动脚本。
- **尚未在 IsaacLab 中实际导入/构造此配置，也未执行完整环境 GPU 步进。**
  当前 CPU 环境没有 IsaacLab；合成测试不等于配置运行或物理接触验证。
- 没有获验收的 HUD04 参考动作、没有 PPO、没有技能学习结果；`training_ready=false`。
- 不创建 Flux 任务，不调用求解器，不在线安装依赖，没有自动训练/重试。
- 已完成的 `TASK_20261116_732` 和 `TASK_20261117_016` 不应重新提交。

## 文件与行为

`overlay/instinctlab/` 下六个文件将新增到原项目的
`source/instinctlab/instinctlab/`：

- `assets/hud04.py`：按哈希载入交付仓库中的已验证资产配置。浮动基座、29 身体关节，
  头部 yaw/pitch 结构性 fixed；参考机器人关闭碰撞和重力。
- `assets/hud04_contract.py`：纯 CPU 的模型、参考动作及版本检查。
- `tasks/shadowing/perceptive/config/hud04/__init__.py`：注册两个新任务。
- 同目录 `perceptive_shadowing_cfg.py`：继承通用感知环境，不继承 G1 环境；
  显式使用 `base_link` 根、`waist_pitch_link` 胸口、HUD04 关键连杆与末端。
- `agents/__init__.py` 和 `agents/instinct_rl_ppo_cfg.py`：独立 HUD04 runner 配置。

任务名：

```text
Instinct-Perceptive-Shadowing-HUD04-Plane-v0
Instinct-Perceptive-Shadowing-HUD04-Plane-Play-v0
```

首轮设置：2 个环境、50 Hz 控制、200 Hz 物理；动作保持运行时关节顺序，
按名称缩放并加 default offset，`last_action` 保持原始动作。
保留通用 policy/critic 观测及 8 帧本体历史，修正关节与关键连杆选择。
相机使用上游 grouped/noisy 管线的归一化和 18×32 裁剪缩放，
而不是把原始 27×48 探针输出直接交给策略。

首轮关闭质心、质量、默认姿态、执行器、材质和相机位姿随机化，以及自适应动作采样。
Plane 接触规则暂只允许足部支撑，手部不作为允许接触；地形越界终止关闭。
这些是待验证的首轮集成设置，不是已经调好的 HUD04 奖励/控制参数。
后续攀爬等手部接触技能必须重新配置接触规则。

## 安装与 CPU 检查（以下命令不启动 GPU）

在自己的工作目录准备两个 checkout。下面变量填写自己机器上的实际路径，
不是 ARIA 沙箱路径：

```sh
export HUD04_DELIVERY_ROOT=/your/workdir/ins1116
export INSTINCTLAB_ROOT=/your/workdir/InstinctLab
export HUMANOID_DESCRIPTION_DIR="$HUD04_DELIVERY_ROOT/flux/hud04_asset_camera_probe/humanoid-description"
```

原项目应处于上述固定提交；不要为切换版本而覆盖自己的未提交工作。
HUD04 模型必须是交付仓库已准备的
`HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf`，不是原始可动头部模型。

```sh
cd "$HUD04_DELIVERY_ROOT/integrations/instinctlab_hud04"
python -B -m unittest -v test_cpu.py
python -B preflight.py --check-model
python -B install_overlay.py --upstream "$INSTINCTLAB_ROOT"
# 检查上一步列出的六个新增路径后，显式应用：
python -B install_overlay.py --upstream "$INSTINCTLAB_ROOT" --apply
```

安装器检查 Git HEAD 和三个关键上游文件的 SHA256，编译新增源码，拒绝符号链接、
路径逃逸或覆盖已有不同文件。已有完全相同文件则只校验，不重写。
它不会克隆仓库、修改上游 G1 文件、运行 pip 或启动仿真。
如旧候选文件已存在且不同，应先人工检查差异；不要使用强制覆盖规避检查。

## 参考动作：默认拒绝，不能拿 G1 数据改名冒充

真实环境构造要求显式设置 `HUD04_REFERENCE_MANIFEST`。本交付**不提供伪造的已验收动作**。
使用专用数据目录，只能包含清单列出的 `*retargeted.npz` / `*retargetted.npz`，
不能混入未审查 NPZ、原始 AMASS 或对象/pickle 数组。

清单结构如下。这里故意保留 `candidate`，因此不会通过正式加载门禁：

```json
{
  "schema_version": 1,
  "robot": "HUD04",
  "root_link": "base_link",
  "quaternion_order": "wxyz",
  "angle_unit": "rad",
  "position_unit": "m",
  "model_sha256": "实际固定头部模型的SHA256",
  "status": "candidate",
  "review_evidence": {
    "constraints": "已恢复的动作与求解约束记录",
    "root_frame": "根坐标转换及单位审查记录",
    "fk_contact_replay": "FK/接触/穿地/滑动回放验收证据"
  },
  "files": [{"path": "motion_retargeted.npz", "sha256": "实际数据文件的SHA256"}]
}
```

只有完成对应审查后，才能将状态标记为 `accepted_for_plane_replay`；
这仍不是运动学门禁自动证明了动力学可行，更不是训练预算授权。
每个 NPZ 需要 `framerate`、Unicode 向量 `joint_names`、
`joint_pos[N,29]`、`base_pos_w[N,3]`、WXYZ 的 `base_quat_w[N,4]`。
加载前检查唯一关节集合、有限值、张量形状、单位四元数、位置/速度限位和文件校验和。
按关节名称重排，不裁剪、不缩放、不改动参考数据。

```sh
python -B preflight.py --reference-manifest /your/reviewed/dataset/manifest.json
export HUD04_REFERENCE_MANIFEST=/your/reviewed/dataset/manifest.json
# 明确选择仿真近似相机；不是实机标定结果：
export HUD04_CAMERA_PROFILE=sim_chest_35deg_v1
```

实机标定应使用已有 camera_mount.py 的 manual 接口，不能把头部相机位姿直接当胸口外参。
当前未恢复到可以验收的参考动作与全部历史求解约束；本次没有重新求解或处理旧动作候选。

## 下一阶段验收顺序

1. 恢复并审查 HUD04 参考动作清单，先做 CPU 检查，不新增求解预算。
2. 在兼容 GPU 环境验证真实配置导入/构造、Articulation 根路径、参考加载/重置、
   policy/critic 张量形状、相机/18×32 输入、多环境隔离和正常清理退出。
3. 新诊断应记录地面过滤足部接触、逐关节误差/力矩、根速度和姿态；
   现有 2 秒浮动探针已有漂移，不可宣称稳定站立。
4. 通过完整环境验证且确认预算后，再运行实际 PPO 更新、checkpoint 保存/重载。
   runner 默认仅 1 iteration、1 minibatch、1 learning epoch，禁止自动 resume；
   这些保守默认值仍不是执行训练授权或硬费用上限。
5. 再接 TerrainMotion 的匹配 metadata.yaml、网格和技能数据，做成功率/泛化评估。

当前不提供一键 GPU/长训练命令，防止跳过数据验收或重复提交已完成任务。
W&B 指定项目此前返回 audience 权限错误；本次没有创建新的 W&B run/artifact，
后续执行前需要恢复该项目的实际可写日志通路。

## 2026-11-17：已调试动作文件夹的实际兼容性审查

来源：<https://drive.google.com/drive/folders/11FgL1RjM32bNNR1d1M1_Wgfntz7Hxtst?usp=sharing>。
已读取其中全部四个 NPZ、两个 YAML 和一个二进制 STL，保留原文件不变。
源数据未上传到本仓库；`MOTION_REVIEW_20261117.json` 记录 SHA256 和结果。

**可继续复用为源动作候选，但四段均不能原样作为当前 HUD04 已验收参考加载。**
29 个关节名称匹配且无头部通道，并不代表两种机器人运动学、限位及根坐标相同。
文件本身没有足够的源机器人版本和调试验收证据，不能仅凭文件名断言是 G1 或 HUD04。

| 源文件简名 | 帧数（90 Hz） | 至少一关节位置超限的帧数 | 最大超限角度 | 至少一关节超速的相邻帧区间数 |
|---|---:|---:|---:|---:|
| jumpsit2 | 967 | 484 | 31.32° | 106 |
| kneelClimbStep1 | 997 | 261 | 37.52° | 165 |
| rollVault11 | 942 | 585 | 47.52° | 131 |
| superheroLanding | 522 | 115 | 37.52° | 100 |

位置/速度对比采用当前固定头部 HUD04 URDF，数值容差 `1e-6`。
速度是 `abs(diff(joint_pos))*fps` 的离散参考值，不是测得的执行器速度。
膝下限有约 `0.000267 rad` 的微小舍入差异，但结论并不依赖它：
爬台阶动作左膝上限实测为 `2.879800 rad`，HUD04 为 `2.583 rad`；
右腕 yaw 最小值为 `-1.439908 rad`，HUD04 下限为 `-0.785 rad`。
不能靠放宽容差、直接裁剪、降低帧率或放宽模型限位来宣称验收通过。

引用关系也有差异：

- `metadata.yaml` 引用缺失的 `kneelClimbStep6-x-0.2-ziwen-retargeted.npz`，不可直接使用。
- `metadata_kneelClimbStep1.yaml` 的动作与网格文件确实齐全，可作为该源动作的配对元数据。
  这只说明引用可解析，不说明运动学、地形单位、接触或动力学已经验收。
- STL 有 8087 个三角面；单位、坐标对齐和物理碰撞尚未验收，不能按文件名假设尺寸。
- 攀爬动作不应塞进当前只允许足部接触的 Plane 环境；需要配套 Terrain 与接触配置。

新增只读检查工具，不调用网络、GPU、求解器，不生成 accepted 清单、不修改数据：

```sh
python -B -m unittest -v test_cpu.py test_motion_audit.py
python -B audit_motion.py --dataset /path/to/downloaded/folder \
  --urdf "$HUMANOID_DESCRIPTION_DIR/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf" \
  --report /path/to/new_audit_report.json
```

报告路径必须尚不存在。当前合计 **44 项 CPU 软件测试通过**，其中八项是新增审查器测试；
不是仿真或参考动作验收。

下一步先找回这段动作针对 HUD04 调试后的导出文件、源机器人/根坐标约定及验收记录，
按 SHA256 排除拿到的是旧版。若当前文件确实是唯一版本，则先约定有界的 HUD04 重定向
与接触修复方案及求解预算，再生成新候选，不默认续用历史 LP/QP 配额。
目前未启动任何新的 GPU 任务、求解或 PPO，也没有修改原有参考门禁。

## TerrainMotion 接入前的只读契约检查

新增 `audit_terrain_bundle.py`，不安装/导入 IsaacLab，不调用求解器，也不注册新的
Terrain 任务。它将加载器静态兼容性与动作数值验收分开报告，不能替代参考动作门禁。

对固定上游提交的 `terrain_motion.py::match_scene` 与
`mesh_terrains.py::motion_matched_terrain` 的源码核对发现：

- 网格按 `terrains` 列表位置选择，场景来源匹配也使用该位置作为字典键。
  因此本版本要求按列表顺序使用整数 `terrain_id: 0, 1, ...`。
  仅检查动作与地形 ID 能互相引用还不够；不能照搬配置注释里的任意字符串 ID。
- 动作采样权重必须有限、非负且总量为正；工具拒绝布尔值和数字字符串，避免隐式转换。
- 网格先按 `[-size[0]/2,size[0]/2] × [-size[1]/2,size[1]/2]` 裁剪，随后根据边缘高度
  进行平移。不能只加载原 STL 后宣称场景中保持了全部地形或原点对齐。
- 通用感知环境的原始 tile size 是 `(9,12)`。本工具必须显式传入尺寸，
  不将它默认为 HUD04 已验收地形尺寸，也不自动调整它。

```sh
python -B -m unittest -v test_cpu.py test_motion_audit.py test_terrain_bundle_audit.py
python -B audit_terrain_bundle.py --dataset /path/to/downloaded/folder \
  --metadata metadata_kneelClimbStep1.yaml \
  --urdf "$HUMANOID_DESCRIPTION_DIR/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf" \
  --tile-size 9 12 --report /path/to/new_terrain_audit.json
```

对已提供 Step1 配对数据的检查：静态文件/ID/权重契约通过，但动作数值门禁仍失败。
假定原始数据使用同一坐标与单位，8087 个三角面中有 59 个含裁剪框外顶点，
其中 33 个完全位于同一裁剪平面之外；997 帧根位置的 XY 均在框内。
根位置在框内不说明手足没有越界，更不证明接触成立。这些计数不是精确裁剪后网格，
未计算边缘高度偏移，也未证明地形单位、根坐标或物理回放兼容。

`TERRAIN_REVIEW_20261117.json` 记录配对文件哈希、上游来源和上述结果；
七个源文件哈希再次全部一致，无数据重写。默认 `metadata.yaml` 仍引用缺失的 Step6，
工具不会代为改成 Step1。现有 Plane 只允许足部支撑，不应直接加载攀爬数据。

截至 Terrain 检查阶段，合计 **58 项 CPU 软件测试通过**（含 14 项新 Terrain 检查测试）。
没有提交 GPU、调用求解器、运行 PPO 或改变 `accepted` 状态；原有六文件 overlay 和
已验证探针不变。下一步仍需确认原调试模型、根坐标及既有约束；若必须新做重定向，
先约定有界求解方案/预算，而不是以本工具通过作为自动开跑许可。

## G1 → HUD04 重定向准备入口（2026-11-17，纯 CPU 已执行）

源机器人家族已确认为 G1；具体导出模型、NPZ 根坐标和四元数顺序仍未得到来源证明。
新增 `retarget_dry_run.py` 和 `kinematic_diagnostics.py`，只执行确定性 FK 与原始网格探测，
**不存在 `--solve` 或 GPU 执行模式**，不构造优化器、不生成动作、不改已有验收门禁。

- 严格固定候选 G1 URDF、HUD04 固定头 URDF、Step1 NPZ 和配对 STL 的 SHA256；
  任一字节不同即拒绝。执行前后都核对输入哈希，输出报告拒绝覆盖已有文件。
- G1 候选模型来自上述上游固定提交的
  `assets/resources/unitree_g1/urdf/g1_29dof_torsobase_simplified.urdf`。
  此文件是诊断候选，不宣称就是生成此 NPZ 的模型。
- 在全部 997 帧计算源 FK；对三个独立窗口 611–641、659–689、711–741（零起始、含端点）
  的 93 帧，分别评估 `torso_link`/`pelvis` × `wxyz`/`xyzw` 四种条件假设。
  不自动排序、选择假设或把几何外观当作来源证明。
- 足部取真实碰撞盒的八个角点，计算到原始 STL 上方包络的垂向间隙；速度仅使用各窗口
  内的实际相邻帧，不跨窗口做差。这不是完整碰撞距离、实际穿透深度或支撑标签。
  原始网格与世界坐标/单位尚未校准，阈值明确以原始单位记录。
- 手部仅报告腕末端原点范围，不把腕原点称为校准后的掌心接触点。
  HUD04 默认姿态的足盒/末端几何单独报告，不复制 G1 的关节角或根位姿作为目标解。

实测：候选 G1 模型的胸部→骨盆变换相对首帧最大变化为 **43.3298°**，
平移最大变化为 **0.0170164**（URDF 长度单位）。因此固定偏移的根坐标转换不可用。
以独立的标量 SciPy FK 复核 5 个真实帧的全部连杆，最大矩阵元素误差小于 `1e-14`；
另对全部 997 帧腰部链、世界坐标重新锚定及 8 个真实网格探测点进行了独立公式复核。

运行时设置好前文的 `HUD04_DELIVERY_ROOT`；以下输入变量指用户保存的原始文件，
脚本不下载模型或数据、不安装依赖：

```sh
python -B -m unittest -v test_cpu.py test_motion_audit.py test_terrain_bundle_audit.py test_retarget_dry_run.py
python -B retarget_dry_run.py \
  --source-urdf "$G1_CANDIDATE_URDF" \
  --target-urdf "$HUMANOID_DESCRIPTION_DIR/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf" \
  --motion "$MOTION_DIR/kneelClimbStep1-x-0.1-ziwen-retargeted.npz" \
  --terrain "$MOTION_DIR/50cmboxes_19_36_49.stl" \
  --report "$NEW_RETARGET_DIAGNOSTIC_JSON"
```

合计 **90 项 CPU 软件测试通过**，其中新入口/几何测试 32 项。七个原始数据文件哈希不变。
`RETARGET_DRY_RUN_REVIEW_20261117.json` 保存结果摘要；不发布原始 NPZ/STL。
本次无 GPU 提交、无求解、无 PPO，`accepted_reference=false`、`training_ready=false`。

下一阶段应补足来源/坐标证据与手/足/膝接触审查，再实现带进程超时监督的有界求解。
报告中的 93 帧、每窗口 90 秒、总计 600 秒、零重试等值只是求解预算记录，
本入口不运行求解，也不能用这些字段冒充已实现的进程资源控制。
短时 GPU 许可不能替代上述技术前提，更不能被解释为长训练或无限重试预算。

## 实际上游任务模型与接触几何补核（2026-11-17，纯 CPU）

**重要区分：上一节的 simplified URDF 是条件诊断候选，不是已证明的 NPZ 导出模型，
也不是固定上游 G1 感知任务实际配置的模型。不要把上一节足盒间隙当作本任务接触结果。**

本次沿实际配置路径核实：`G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG`，
`robot_model_path = G1_CFG.spawn.asset_path`；物理资产和参考 FK 都使用
`g1_29dof_torsobase_popsicle.urdf`，SHA256 为
`5e99d930af64f3f0bfd34235239e288cc380ba62c8b01fcfcf8cc9f0996e7fa4`。
该 URDF 根是 `torso_link`；NPZ 加载器直接使用四元数字段，按 IsaacLab 的 `wxyz`
约定处理。**这些是已核实的运行时解释，不足以证明所提供 NPZ 当初按此约定导出。**

新增 `audit_source_contact_model.py` 与 `contact_geometry.py`：

- 在全部 997 帧比较实际任务模型与早先候选的 35 个共有连杆：同名连杆原点
  最大相差 **19 mm**，足/膝原点最大相差约 **12.05 mm**，腕原点相差 **10 mm**。
- 实际 G1 每只脚有 **7 个圆柱碰撞体**，且资产配置开启圆柱转胶囊；HUD04 则明确关闭
  此转换。已核实 Isaac Sim 5.1.0 固定提交
  `47d886f2858d1ceed556b21c88927aa67bc81c12` 的 `UrdfImporter.cpp`：
  胶囊保留圆柱的轴段长度，再加半球端帽，不能按原始圆柱包围盒计算轴向端点。
- G1 的手部胶囊中心在 wrist-yaw 连杆局部 `[0.075, 0, 0]`；HUD04 的实际手网格在
  yaw→pitch→roll 链的末端 wrist-roll 连杆。不能把同名 wrist-yaw 原点直接对齐为掌心。
  新工具读取并固定两侧手网格哈希，支持盒、球、圆柱、胶囊及网格顶点的方向支撑点。
- HUD04 足盒局部边界是 `[-.107,-.047,-.0635]` 至 `[.143,.047,-.0435]`。
  现有 `contact_foot_tip_L/R` 的 x=.154，比碰撞盒前缘多出 **11 mm**；三个现有足标记
  z=-.0625，均比盒底高 **1 mm**。center 标记也比盒底几何中心向后 **18 mm**。
  **这些标记没有被修改或自动接受为支撑目标**；它们可能服务于原始模型的不同用途。

独立复核采用闭式投影包围盒、`struct` STL 解析和 SciPy 标量 FK：14 个几何连杆、
两个源模型的全部 997 帧均交叉检查；包围盒最大误差 `4.67e-15`，FK 最大矩阵元素误差
`1.12e-15`。合计 **113 项 CPU 软件测试通过**，本阶段新增 23 项。

工具仍是只读 preflight；参数只接收本地固定哈希输入与一个不存在的 JSON 报告路径，
没有求解/回放/训练模式。运行示例（沿用上文的本地输入变量）：

```sh
python -B -m unittest discover -p 'test_*.py'
python -B audit_source_contact_model.py \
  --task-urdf "$G1_POPSICLE_TASK_URDF" \
  --earlier-urdf "$G1_CANDIDATE_URDF" \
  --target-urdf "$HUMANOID_DESCRIPTION_DIR/HU_D04_description/urdf/HU_D04_01_fixed_head_body_collision.urdf" \
  --motion "$MOTION_DIR/kneelClimbStep1-x-0.1-ziwen-retargeted.npz" \
  --report "$NEW_SOURCE_CONTACT_MODEL_JSON"
```

`SOURCE_CONTACT_MODEL_REVIEW_20261117.json` 保存摘要与来源。几何方向支撑/包围盒不是
物理接触标签、网格完整碰撞距离或校准后的掌心坐标；未验证 PhysX 烘焙结果、接触裕量
或地形世界对齐。七个源数据文件、既有探针、HUD04 资产和 overlay 均未改动；
本阶段 GPU/求解/PPO 全为 0，`accepted_reference=false`、`training_ready=false`。

下一步按实际上游任务模型重建有明确假设标记的足/手/膝—地形关系，
补齐导出/坐标证据后再决定接触约束与有界重定向；不自动选择 NPZ 来源解释，
不启动已有 GPU 任务，也不把本 preflight 当作站立或攀爬成功证明。

## 正确 G1 几何下的条件式窗口重建（2026-11-17）

新增 `audit_conditional_contact_windows.py`。沿用已审查的 popsicle 模型与圆柱转胶囊
规则，在 611–641、659–689、711–741 三个独立窗口（零起始、含端点，共 93 帧），
分别计算 `torso_link`/`pelvis` × `wxyz`/`xyzw` 四种明确标记、**不排序、不选择**的假设。
每帧对每个真实名义胶囊取一个沿世界负 z 的支撑极值点：每侧足 7 个、腕 1 个、膝 1 个。
每假设 1674 点，共 6696 点；不是按点数累计出的接触帧数。

```sh
python -B -m unittest discover -p 'test_*.py'
python -B audit_conditional_contact_windows.py \
  --source-urdf "$G1_POPSICLE_TASK_URDF" \
  --motion "$MOTION_DIR/kneelClimbStep1-x-0.1-ziwen-retargeted.npz" \
  --terrain "$MOTION_DIR/50cmboxes_19_36_49.stl" \
  --report "$NEW_CONDITIONAL_CONTACT_JSON"
```

源 URDF/NPZ/STL 均固定 SHA256、运行前后复核；输出拒绝覆盖。报告给出原始 STL
上方 XY 包络的垂向间隙分布、无高度交点数和明确以原始单位表述的阈值计数。
独立解析胶囊极值与另一套三维平面/重心坐标射线求交公式已复核全部 6696 点，
最大支撑点误差 `4.45e-16`，高度误差 `2.17e-19`，阈值计数全部一致。
合计 **123 项 CPU 测试通过**（新增 10 项），七个原始文件哈希不变。
`CONDITIONAL_CONTACT_WINDOWS_REVIEW_20261117.json` 保存精简证据。

**这些间隙不是完整碰撞距离或支撑标签。** 原始世界单位、根/四元数来源未证明，
尚未应用实际地形裁剪/边缘高度平移；垂直墙、侧向手掌支撑及悬垂不能由上方包络判定。
某假设低于包络的点少，不表示它正确或更适合训练。极值点随姿态可以换到不同材料点，
因此不对极值轨迹做差并冒充足/手滑移速度。工具不生成动作/接触标签，不调用求解器
或 GPU，不改变任何门禁。下一步仍需坐标证据与接触面/支撑区间审查，再接入有界修复。