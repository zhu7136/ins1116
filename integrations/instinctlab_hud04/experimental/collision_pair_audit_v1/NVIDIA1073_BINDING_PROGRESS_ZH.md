# NVIDIA107.3文档绑定进展（2026-11-18）

基线提交84ce0a867c23cee03821f3a214346a48b9bfb90b。本次授权仅用于读取NVIDIA文档；没有GPU/PPO/动作优化器调用，也没有改机器人、过滤、接触阈值或旧候选参考。

## 已从官方107.3文档确认

- `IPhysxSimulation.subscribe_contact_report_events`接收双buffer回调，返回subscription holder；contact buffer仅在一个simulation step内有效。三buffer（含friction anchors）的API名是`subscribe_full_contact_report_events`，不能将新版不同名称的API直接当作兼容回退。
- `ContactEventHeader`包含actor0/1、collider0/1的uint64 ID，文档给出的解码器是`PhysicsSchemaTools::intToSdfPath`；Python封装显式检查调用方提供的`intToSdfPath`是否可调用，不宣称已在目标镜像导入成功。header还包含stage_id和proto_index0/1；`0xFFFFFFFF`表示非point-instanced collider。
- `ContactData.separation`确有字段，文档称contact separation value。此条目没有充分说明符号、单位与offset校正，不升级为几何穿透深度。
- `RigidContactView.get_contact_data(dt)`逐项列出六个数组，与上一提交固定源码一致；dt用于impulse到force的转换，不能再除一次dt。官方摘要写five但逐项列six，是文字不一致，读取器仍严格要求六项。
- `PhysXCooking.request_convex_collision_representation(stage_id, collision_prim_id, run_asynchronously, on_result)`是公开API，返回task，回调获得result和convex mesh列表。mesh具有vertices/indices/polygons，polygon具有index_base/num_vertices/plane。**该描述不足以保证这是当前求解器live shape的被动快照**；不能直接代替同stage/形状、缩放/坐标系、cooked子形状及运行时过滤等价性证明。本轮没有调用它或请求烹饪任务。

引用及下载HTML的SHA256保存在`NVIDIA1073_API_EVIDENCE.json`，含8个精确API锚点和摘录。哈希是本次获取内容的快照哈希，不是上游不可变revision。旧`API_SOURCE_EVIDENCE.json`保留历史事实，其“文档尚不可访问”状态已被本文件与新增证据更新，不重写历史文件。

另已读官方[107.3碰撞指南](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html)，本次137562字节HTML快照SHA256为`a57ee3f730a9ea50bc955ee060f1577ae5dbad57542d0a68c293a75864709a5c`。`#create-a-cylinder-collider`明确：`SETTING_COLLISION_APPROXIMATE_CYLINDERS=True`会将圆柱近似为凸网格，近似后的圆柱忽略margin属性。因此**原始USD仍为Cylinder不代表实际求解器一定用解析圆柱**，后续必须读回实际设置和cooked表示，不以原XML/Cylinder元数据代替。`#configure-rest-and-contact-offsets`区分contact生成距离与rest offset，后者可正/零/负；有contact不等于原几何已经相交。`#generate-mesh-colliders-cooking`说明缓存缺失会触发计算并阻塞初始化；不在生产任务中盲目触发cooking或改缓存设置。本轮仅查文档，没有设置上述选项。

## 新增实现与CPU测试边界

`contact_report_1073.py`复用原严格observer，新增：

1. 必须由调用方独立提供非零expected stage ID；检查每个header的stage匹配，避免同路径跨stage误绑定。
2. 只接受非instanced header；不支持的proto index、缺失字段、非法uint64、未知解码ID、错误callback参数数目和超预算均留存错误，不抛入仿真循环，也不把未知当零接触。
3. 在回调有效期内复制有效接触；保留原始ID、stage/实例索引与原始separation。回调不写USD、不设置报告API/threshold、不推进仿真。
4. 构造器不订阅。只有显式`subscribe_existing(existing_interface)`才调用已存在接口的文档方法，并返回holder由调用方保活。该方法不导入/创建仿真，不静默回退新版方法，不隐式重复订阅；失败后不自动重试。**本轮仅用fake interface单测，不是真实订阅验证。**

独立stage ID取得方式、真实完整镜像版本、native接口存在性、报告enable/threshold与有效时间覆盖仍需目标进程中验证，不能由CPU参数或成功返回替代。真实接入时应在原任务已经运行的同一articulation/同stage中进行，并保存holder生命周期证据，不另建分离副本替代因果对照。

命令：

```sh
python -m unittest -v test_audit test_native_contact_observer test_tensor_contact_reader test_contact_report_1073
```

实际结果66项通过（原50+新增16），0.046秒。记录见`TEST_NVIDIA1073_EXECUTION.log`。数据来自历史审计fixture与合成buffer；不是新的物理实测。

## 尚未变化的验收状态与下一步

- 冻结FULL997候选仍为4692条严格相交、277条未决；不接受为训练参考，`accepted_reference=false`、`training_ready=false`。
- 新增GPU任务0、PPO0、优化器调用0、自动碰撞豁免0；本轮没有创建W&B run/artifact。
- 下一步先完成同原articulation、精确stage/shape绑定、最终过滤状态和cooked几何能力读回的最小观测规格。需实测才能闭合的项保持UNRESOLVED，不靠放宽阈值或删除碰撞约束处理。
- 延续既有不新增GPU/PPO边界；网络授权不是GPU授权。需另行明确授权与有限预算后，才提交新的最小运行时观测；不重跑固定基座、退出诊断、分离副本任务或旧LP窗口。