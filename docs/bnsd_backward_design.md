# Flash Attention 3/4 反向布局与 stride 设计

## 接口契约（2026-10-26 更新）

仅修改 FA3/FA4 在 910、950 上的反向实现，前向函数、公共签名和前向 autograd context 不变；验收可以直接调用反向。本文替代此前要求调用方传 `layout=2`、返回物理 BNSD 梯度的方案。

- dense 输入的逻辑形状始终为 BSND：q=[B,Sq,Nq,Dqk]，k=[B,Sk,Nkv,Dqk]，v=[B,Sk,Nkv,Dv]，out/dout=[B,Sq,Nq,Dv]。输入可以是连续 BSND，也可以是连续 BNSD 存储的 `transpose(1,2)` 视图。
- Python 复用前向 `_detect_layout`，按 stride 在内部识别 BNSD；有 cu_seqlens_q 时选择 TND。Python backward wrapper/custom op/fake 不暴露 layout 参数。
- Host 接收内部布局编码（0=TND、1=BSND、2=BNSD），转换成对应 kernel 模板。dense shape 统一按逻辑 BSND 解析，不能因为物理 BNSD 而交换 head/sequence 轴。
- dq/dk/dv 为连续 BSND，与 q/k/v 的逻辑形状一致；TND 输入返回连续 TND。预分配梯度 buffer 必须满足该约束。
- dense LSE 仍为 float32 [B,Nq,Sq]；packed LSE 保持已有 [Nq,Tq] 契约。

与前向相同，连续四维张量的 shape 无法表达它是否采用 BNSD 语义。BNSD 存储必须以逻辑 BSND 视图传入，不能把连续 `[B,N,S,D]` 直接作为 `[B,S,N,D]` 输入。

## 泛化寻址

新增共享 POD `FagTensorStrides` 和 `FagInputStrides`，通过 tiling 传递 q/k/v/out/dout 各自的 batch、sequence、head 元素 stride；保留原 tensor 的 `data_ptr()`，因此支持非零 storage offset。

输入统一按以下公式寻址：

```
offset = batch * batchStride + token * sequenceStride + head * headStride
```

TND 的 batchStride=0，token 使用 cu 前缀加当前 tile 的 token 起点。D 维保持连续；token 行不能相互重叠。910 保留 Dqk=Dv 的既有约束；950 保留独立的 Dqk/Dv。标准 BSND/BNSD/TND、带 token/head/batch 间隙的切片、不同输入物理布局组合都使用实际 stride，而非从另一个 tensor 的偏移除以 D 后推导地址。D 不连续的输入继续沿用 wrapper 的连续化处理。

识别规则与前向一致：只有 q/k/v 同时符合前向的 BNSD stride 模式才选择 BNSD 模板；其他 dense 选择 BSND 模板。无论选哪个 dense 模板，kernel 均读取真实 stride，所以混合布局及带 padding 的视图也能正确寻址。

## 910

- Cube QK/dOV、dQ/dK/dV 的输入分别采用对应 tensor 的真实 stride。
- 输入地址和梯度 workspace 地址分开：累计 workspace 始终采用 BSND/TND，不把 BNSD 输入偏移用于梯度写回。
- Sfmg 分别按 out/dout stride 搬运；中间 delta 保留原 BNS 行顺序。共享 Sfmg 对旧 FA2 保留 canonical stride 默认值。
- deterministic reducer 的 dense 输出按 BSND 处理；post 仍线性 scale/cast，不增加全量 transpose。

## 950

- Cube input tensor 的 GM offset/row stride 从各自的 stride 描述计算；梯度 workspace 和最终输出保持 BSND/TND。
- Sfmg 按 BNS（packed 为 N,T）遍历，每次搬运限制在当前 head plane 内，分别使用 out/dout 的 token stride；delta 存储和 SubMul 读取采用相同顺序。
- BN2S2 的调度坐标、梯度 workspace 和 private buffer 与输入布局解耦，dense BNSD 可以复用已有调度。各版本既有 deterministic shape 覆盖限制保持不变。
- 不改变 Cube tile、UB buffer、事件、跨核同步和归约顺序。

## 调用示例

```python
# 物理 BNSD，向接口呈现逻辑 BSND。
q = q_bnsd.transpose(1, 2)
k = k_bnsd.transpose(1, 2)
v = v_bnsd.transpose(1, 2)
# out/dout 也用逻辑 BSND，允许物理 BSND 或 BNSD 视图。

# FA3：梯度写入预分配的连续 BSND buffer。
dq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
dk = torch.empty(k.shape, dtype=k.dtype, device=k.device)
dv = torch.empty(v.shape, dtype=v.dtype, device=v.device)
_flash_attn_backward(dout, q, k, v, out, lse, dq=dq, dk=dk, dv=dv)

# FA4：分配并返回连续 BSND 梯度。
dq, dk, dv = _flash_attn_backward(q, k, v, out, dout, lse)
```

本次保持前向保存张量和 autograd context 原样；直接反向接口使用上述逻辑 BSND 契约。本文不把前向 context 中已保存的物理 BNSD 张量等同于逻辑 BSND 视图，也不声称已修改该 autograd 联动。

## 验证

- CPU 测试加载实际 wrapper、检测函数和 fake：覆盖 BSND/BNSD/TND/padded，检查自动分发、无输入拷贝、连续梯度 buffer 和 fake 输出逻辑形状。
- NPU 直接反向测试使用 CPU float64 reference 提供 out/LSE/golden，不调用本仓前向。覆盖 FP16/BF16、MHA/GQA/MQA、矩形/尾块、causal/local(910)/softcap、deterministic、storage offset、混合输入布局、BSND/BNSD token padding 和 TND token padding；校验输出是连续 BSND/TND。
- 编译 910/950 的 FA3/FA4，比较生成文件、Python 语法及 diff；AST/源码比较确保前向和公共签名不变。
- 宿主机已由用户确认可以枚举 8 张 Ascend950PR；代理执行环境看不到设备节点，torch_npu 无法枚举设备。此限制不能解释为宿主机没有 NPU；编译和 CPU 检查不能代替上板精度、确定性与性能验收。

```bash
python -m pytest tests/test_bnsd_backward_metadata.py -q
python -m pytest tests/test_flash_attn_bnsd_backward.py -q
```

## 本次验证记录

2026-10-26，CANN 9.2.0-beta.2，CATLASS 锁定提交 `769cd40a8716b28650b6bebb08db4834eea4462f`：

- FA3/FA4 × 910/950 四个扩展均完成编译和链接；950 最后一次输入 shape 校验变更也已增量构建。
- 修改涉及共享 Sfmg，额外完成 FA2 910 编译回归。
- CPU 接口/元数据检查：32 passed；四个 custom op 的 schema 推导通过，均不含 layout 参数。
- 直接反向 NPU 测试模块因代理执行环境无法访问宿主机设备而跳过；新增用例尚未完成上板精度及确定性验收。
- Python 语法、dispatch 生成一致性及 `git diff --check` 通过；源码/AST 比较确认前向、公共签名及 autograd 类保持不变。

## 上板反馈与 UB 修正（2026-10-26）

用户运行直接反向测试时，前 44 项通过；随后 FA3/950、FP16、deterministic、D=256 在 host tiling 阶段报 UB 超限。旧逻辑固定 cast 32 行：pipeline 222208 B + cast 49152 B = 271360 B，超过实测 UB 253952 B。

现由 host 根据剩余 UB、Dqk/Dv 和 32B 对齐计算 cast chunk，并通过 `detCastChunkRows` 传给 kernel。该场景选择 20 行：FP32 输入 20480 B + 16-bit 输出 10240 B，总峰值 252928 B，pipeline 与 cast tail 之间仍有 1024 B 空隙。空间连一行都不够时仍拒绝，不移除容量检查，不改变调度、归约顺序或精度阈值。FA3/FA4 共享该 950 实现。

新增 `tests/test_fag_cast_ub_plan.py`，直接编译并运行生产使用的 host planner，覆盖 D=192/256、独立 value dim、容量边界与对齐等 20 项，全部通过。测试 reference 继续在 CPU float64 计算，out/LSE 在 CPU 转为目标 dtype 后才搬到 NPU，避免设备 double 转换警告。

UB 修正后的上板精度及确定性需重新运行原测试确认；用户给出的 44 passed 不能视为修改后全集通过。

本轮 UB 修正验证：20 项 host planner 回归通过；FA3/FA4 两个 950 扩展重新编译、链接并原地更新完成。尚未取得修正后的 NPU 测试结果，需在可访问设备的终端重跑直接反向测试。

## D>128 确定性梯度写回修正（2026-10-26）

UB 修正后用户再次运行同一组测试，前 44 项通过，随后 FA3/950、FP16、deterministic、D=256 的 dK 对 CPU float64 reference 有 906/66048 项超过 0.003 容差，最大绝对误差约 0.00653。该错误发生在 kernel 数值阶段，与先前的 tiling UB 报错分开处理。

代码审查发现：950 的 D>128 `ComputeDqDkLargeHeadDim` 和 `ComputeDvLargeHeadDim` 把写回模式固定为 atomic-add，未接收调用方传入的 `enAtomicDk`/`enAtomicDv`。确定性 BN2S2 的 dK/dV private buffer 在列的首个贡献时需要覆盖写，之后才累加；固定 atomic-add 可能读入尚未初始化或先前列留下的数据。本次把写回模式贯通到每个 D slice。普通非确定性路径的 D>128 仍显式传 atomic-add，因为它每个 slice/task 写回预清零的全局 workspace；D<=128 继续在 L0C 中积累，到 flush 时写回。

新增 `tests/test_fag_large_dim_atomic.py`，把生产头文件中的四个实际函数体编入主机 harness，检查 D=129/192/256、dK/dV、覆盖与 atomic-add 两种模式共 12 项，均通过。NPU 测试在失败时会分别给出 dQ/dK/dV 对 golden 的超阈值数量、最大绝对误差，以及 BNSD 与 BSND 结果的最大差值；容差未调整。此处的源码证据和主机测试尚不能证明上板精度已恢复，需重新运行原 pytest 命令验收。

本轮验证：950 的 FA3/FA4 扩展重新编译、链接并原地更新成功；CPU 接口/元数据、UB planner 和生产 D>128 写回函数 harness 共 64 项通过，`git diff --check` 通过。

用户在 Ascend950PR 宿主机执行 `/home/wyf/miniconda3/envs/wyf/bin/python -m pytest tests/test_flash_attn_bnsd_backward.py -q -x`，结果为 **76 passed, 4 skipped in 3.44s**。先前失败的 D=256 deterministic FP16 用例现已通过。4 项跳过均为 950 反向暂不支持的 local-window 组合（FA3/FA4 × deterministic 开/关）；910 硬件尚未上板验证，已有对应编译和主机检查结果。
