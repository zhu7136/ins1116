# 原articulation只读运行时观测：准备完成，未启动物理进程

## 1. 固定对象与已知缺口

- 原始基线：`f9924ea919eeeabeb8ddca4e51d8463d0f2fb18c`。新增准备代码以本交付提交固定；不重跑或修改旧探针。
- 沿用历史目标镜像reference：`BJX00000178 / V000446`；历史依据为`measured_task138/TASK133_DUPLICATE_REVIEW.json`中的已结束任务133，清单绑定该文件SHA256。**未核验当前服务镜像内容digest或当前运行镜像，不将版本标签当作不可变镜像。**
- 原始URDF：`HU_D04_01_fixed_head_body_collision.urdf`，SHA256 `865a4344e2f838d939c2549ec0d9e771dc6c2eb745670d96bba78e162d2880d7`。清单复核全部20条既有audit输入哈希，含模型、网格、三阶段快照、旧测量和冻结参考文件。
- 原始生成USD字节/组合依赖层未留存，故**原始USD文件哈希未知**。`post_import/post_clone/post_physics_init.json`只是三份组合USD元数据快照，不是USD文件、live shape或当前配置快照。
- 当前runtime配置、完整build/digest、活跃且获授权的宿主进程均未核验。已有证据不构成当前进程仍存在或可接入的证明。本轮没有启动/连接/重启进程，也没有查询账号全部任务。
- 冻结FULL997参考哈希仍为`2b4070e32e3a5f120caea7942b42e91a6f31bbca9e0b6d7e8e135b50803ad2f4`，4692严格相交/277未决，不做替换或重新验收。

`RUNTIME_PREPARATION_MANIFEST.json`保存哈希和缺失项，执行状态为`WAITING_RUNTIME_IDENTITY_AND_AUTHORIZATION`。这是准备清单，不是当前物理环境快照。

## 2. 交付接口及安全边界

### CPU命令（不导入Isaac/pxr，不创建stage）

```sh
python prepare_runtime_observation.py --prepare NEW_MANIFEST.json
python -m unittest -v test_audit test_native_contact_observer test_tensor_contact_reader test_contact_report_1073 test_runtime_preparation
```

清单输出使用独占创建，已有文件时拒绝覆盖。本次实际写出的清单见上文；不要用命令覆盖历史证据。累计85项CPU测试通过（原66+新增19），最终回归0.074秒；全记录见`TEST_RUNTIME_PREPARATION.log`。fixture和合成buffer不是真实订阅、物理接触或动力学验证。

### 配置采集器：`existing_stage_snapshot`

`prepare_runtime_observation.py`仅接收宿主已经创建的`stage/settings/bindings`与显式精确路径（最多32个）。不扫描整个stage，也不调用`Apply/Set/Load`、不重导入机器人、不获取cooked表示或写物理参数。

- 通过目标bindings中的圆柱/圆锥近似设置常量，调用settings的只读`get`。常量/键不存在、None或类型不符为UNKNOWN，不强制转False。
- 读取指定prim的已组合属性与关系，保留authored、default-time值、time-sample数量、类型和instance proxy状态。应传入精确actor/collider、两腕joint、articulation root、physics scene、已知collision group和相关关系目标。actor上的ContactReport threshold也在原属性名下记录。
- 属性/关系数量、目标数量和元数据数组/嵌套均有限额。不会把Mesh的points/面索引当作live cooked几何读取。default-time元数据不能冒充当前物理姿态；不从缺省元数据猜GPU/CPU管线。
- 未传入的全局group/scene、有效报告启用状态、阈值含义、真正solver最终过滤仍是UNKNOWN。宿主需独立提供CPU/GPU pipeline、stage cache ID与已用层/源文件哈希及来源，再交叉核验；本采集器不伪造这些证据。

### 回调脚本：`ExistingRuntimeObservation`

`observe_existing_runtime.py`复用已验证文档契约的`ContactReport1073`，不是可执行物理场景或任务启动器。

1. 宿主提供原stage的独立ID和`PhysicsSchemaTools`；构造不订阅。明确`subscribe_existing(existing_interface, authorization=...)`才尝试订阅，并且要求已有进程授权receipt、`starts_new_process=False`和非空批准来源。receipt只是调用方声明，仍需人工/服务记录核验，不能凭此自证授权。
2. 返回的subscription holder必须由宿主持有并按宿主已验证的生命周期接口清理。不猜测unsubscribe方法，不关闭宿主进程。收集器本身不保证holder生命周期或订阅时间覆盖。
3. 在宿主**本来就有的**步进前后调用`on_host_step_begin(step)`和`on_host_step_end()`；不额外调用step/reset/play。最多8个连续宿主step、30秒观测时间、256事件、4096点。超限只停止/拒绝记录并报错，不终止或推进物理进程。代码无sleep、轮询、线程或自动重试。
4. 双buffer回调在当前simulation step有效期内立即复制有效数据。每个event保存类型、actor/collider完整路径，每个header保留原始uint64 ID、stage和`0xFFFFFFFF`哨兵；不支持的point instancing明确报错，不当普通实例号。step记录通过event范围关联全部header身份。
5. step未结束、标记跳号、标记外callback、deadline、预算、native字段/ID/布局错误都留存为错误。回调不向仿真循环抛异常。`finish()`冻结记录，迟到callback不再复制；宿主仍负责订阅清理。
6. 每步记录callback次数和缺callback列表；**无callback只是UNKNOWN，不等于证实丢帧或无接触**。native buffer capacity/overflow未公开取得时保持null/UNKNOWN，应用层预算未超不能证明native没有溢出。宿主步标记也不是独立时间覆盖证据。

调用`report()/finish()`返回可JSON序列化结构，宿主可独占创建新日志文件并用`json.dump(..., allow_nan=False)`落盘。采集器不默认上报、创建W&B run或写入既有参考数据。CPU测试只使用fake host/interface；**本轮没有执行上述订阅或宿主物理步骤**。

## 3. 三类独立证据及预期判定表

| 观测/缺失情形 | 允许更新的结论 | 禁止推断/下一步 |
|---|---|---|
| 近似设置、pipeline和exact collider元数据可读 | 分别记录OBSERVED及来源 | Cylinder不能直接判解析圆柱；pipeline、设置与实际shape仍需一致性核验 |
| 源USD文件/依赖层或镜像digest缺失 | 身份缺口UNRESOLVED | JSON快照哈希、镜像版本标签不能替代；新建环境前停止待授权 |
| 关节/shape开关、filteredPairs/group关系可读 | 只更新元数据配置证据 | 不认定已证明求解器最终pair过滤；未遍历的group保持未知 |
| 同stage原actor/collider完整ID出现有效接触报告 | 该精确路径报告过接触，保留原始separation | 不据此证明cooked等价、最大穿透、全程安全或参考通过 |
| 零接触/无callback/阈值不可读 | UNRESOLVED，记录缺失/阈值/覆盖 | 不认定配对过滤、零穿透或无遗漏 |
| ID/stage不符、实例索引不支持、预算或buffer未知 | 错误/不完整或UNKNOWN | 不丢弃坏记录后将其余统计当完整覆盖 |
| 请求凸表示仅返回vertices/faces | 最多记录表示请求结果及其来源 | 不自动称live solver cooked shape；本轮也不执行请求 |
| 有live handle、坐标/尺度/子形状和字节/几何等价证据 | 单独审阅cooked等价性 | 不因此推定最终过滤或参考动作所有验收通过 |

只读证据若不足以判定过滤因果，就保留UNRESOLVED。改变关节过滤开关或做受控接触测试属于**另行授权的物理实验**，不是本只读脚本的回退路径；不得静默修改阈值、collision设置或隔离副本替代原articulation。

## 4. 启用门与验收隔离

当前结论：没有可核验的现存授权宿主，原USD/digest与当前配置缺失，故保持待授权。若确认可接入已有授权的原进程，再核验其身份、资源/租期、同stage路径和holder清理后，只在原循环有限观测；**不因本文件或网络授权而创建新的Isaac/GPU进程**。

如必须新开进程，需用户明确批准有限任务/时长/设备预算及固定的输入签名，再重新核对服务器状态与intent防重提交。当前8步/30秒仅是观测器限制，不是GPU启动预算，更不是训练预算。

即使后续三类证据全部补齐，也只更新表示/过滤/cooked证据。其它自相交、地形相交、接触模式/支撑、浮动基座动力学、控制与段内扫掠验收分别闭环；`accepted_reference=false`和`training_ready=false`绝不自动升级。本轮GPU/PPO/优化器/模型过滤修改/参考替换均为0。