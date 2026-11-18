# 接触分离量读取：固定官方源码依据与CPU适配（2026-11-18）

## 本次新闭合的证据

之前仅有IsaacLab使用六项返回的位置/索引调用，无法据此确认第4项的含义。本轮直接读取并独立校验Isaac Sim **v5.1.0**官方源码：

- 文件：`source/extensions/isaacsim.core.api/python/impl/sensors/rigid_contact_view.py`
- Git blob：`e8d9b2f8a0a1aed3db14fa65d79db1f153598a1f`
- 内容SHA256：`6b48c4c48633a742b794b391275bc8a2274004a1177c882c2b1d057a76004b1b`
- 独立从原始bytes计算Git blob SHA1，与API对象标识一致；v5.1.0 tag下对应文件blob也一致。
- 第350–376行docstring明确提及contact separations；第443–466行将底层`get_contact_data(dt)`第4项命名为`distances`并透传，第5/6项为count/start。

证据与必要源码摘录见`API_SOURCE_EVIDENCE.json`，生成代码`collect_api_source_evidence.py`是一次性只读下载，已执行，已有输出不能覆盖。

**新增结论是字段身份有固定官方源码依据，不是已实测HUD04分离量。** 历史task137失败堆栈中的native tensors包标识为`107.3.26+107.3.3.lx64.r.cp311.u353`，已与证据文件哈希一起保存；这不等于目标镜像全构建或新接口兼容性已被确认，task137失败状态不变。

## 已实现的读取逻辑

`tensor_contact_reader.py`提供：

- `unpack_singleton(...)`：解包已有六项缓冲区；
- `capture_singleton_view(...)`：对**已有原生view**恰好调用一次`get_contact_data(dt=...)`；不创建view/模拟器、不设置姿态、不订阅、不步进、不更改过滤。

严格要求六项shape符合单通道布局：`(C,1), (C,3), (C,3), (C,1), (1,1), (1,1)`。先检查容量再复制到CPU；count/start必须是非负整数，范围不越界，满容量拒绝，不能默默截断。只读取count/start指定的有效接触段，未使用storage中的NaN不算测量；有效段出现NaN/Inf/错误dtype则失败。

报告保存接触法向标量、位置、法向和原生接触分离量，保留符号，不重复除dt，不把力换算成穿透深度。没有接触时最小分离量为null，**不是0，也不是分离/过滤证明**。

期望body路径字段明确叫`expected_*`，不把调用方给出的路径当运行时独立绑定。接口不包含shape-ID，因此仍不能确定原腕部指定cylinder与mesh间的最小距离/最大穿透，更不能将其它同body碰撞形状的接触归给指定pair。

## CPU测试，不是新增物理试验

本轮新增22项测试；与原审计/原生事件接收器测试一起：

```
python -m unittest -v test_audit test_native_contact_observer test_tensor_contact_reader
# 50 tests, OK；实际日志见TEST_TENSOR_EXECUTION.log
```

覆盖非零buffer起点、保留正负分离/力、不二次除dt、未使用NaN区、有效段非有限值、负索引、float/bool计数、多通道形状、越界/饱和、复制与别名、精确路径、dt检查和单次原生读取。测试输入为合成NumPy缓冲区；**没有新的GPU样本**，没有物理结果、优化结果或训练结果可上报。

## 仍需原生文档闭合的事项

本轮发现官方当前分支的碰撞验证示例使用较新的三参数contact callback与不同订阅入口，不能把当前分支方法名直接移植到107.3目标运行时。原生事件接收器仍按通用输入契约保留，不声称已绑定版本正确的订阅API。

该版本差异提示来自本轮只读查看的官方`isaac-sim/IsaacSim`碰撞验证文件固定blob `47dd41b389e04fe87c8cb3feb64ac3642fba5907`，第92–99、132行；可通过GitHub API `/repos/isaac-sim/IsaacSim/git/blobs/47dd41b389e04fe87c8cb3feb64ac3642fba5907`核对。它不是v5.1.0运行时兼容性证据。

需要阅读NVIDIA 107.3文档确认：

1. 该版本真实订阅入口、回调参数个数、事件生命周期、整数ID解码和actor/collider路径含义；
2. separation符号、单位及contact/rest offset的解释，不将接触流稀疏样本冒充全局几何signed-distance查询；
3. 公开API是否可以读取cooked几何及实际shape标识，并能否建立同一原始articulation下的精确对应。

官方文档URL从已核实IsaacLab源码获得：
`https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html`

本轮访问该域名连接被重置，未读取内容；需单独网络授权。未借用GitHub密钥访问NVIDIA域名，也未把第三方旧版stub当官方目标版本证据。

## 不变的边界

GPU提交0、PPO0、运动求解0、碰撞几何/过滤更改0。未执行旧发布/任务脚本，未重跑138拆离探针、固定基座探针或旧LP。`accepted_reference=false`、`training_ready=false`；4692严格相交/277未决的原拒绝状态没有变化。拿到文档网络权限后先完成原生接口绑定，仍不代表获准提交GPU试验。