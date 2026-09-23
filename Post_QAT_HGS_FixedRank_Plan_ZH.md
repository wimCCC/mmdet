# SAR-QAT：训练期量化误差补偿方案

## 1. 工作方向调整

原方案计划在已经训练完成的 QAT checkpoint 上再做一次 Post-QAT HGS/PTQ 补偿。该流程本质上是先完成一次 QAT，再冻结模型、重新校准输入并额外构造低秩分支，不能代表训练期模型实际学习到的量化误差补偿能力，因此不再作为本项目主线。

新的主线是：**在 SSI + LSQ + SRQ 的 QAT 训练过程中直接加入量化误差补偿分支，并让补偿参数参与 QAT 训练。**

HGS 原理只用于：

1. 根据当前量化误差和输入二阶矩初始化补偿方向；
2. 约束补偿分支的低秩结构和插入边界；
3. 提供可选的初始化/诊断信息。

HGS 不再作为训练完成后的第二次 PTQ，也不再单独生成一个脱离 QAT 的后处理插件。

本轮不重新实现 SSI、LSQ 和 SRQ。它们继续沿用当前代码中的 `ClipStdObserver`、`LearnableFakeQuantize` 和 `QDropFakeQuantize`。

## 2. 新的训练流程

```text
浮点权重 Wref → 原有权重量化器 → 量化主分支 Wq ─────┐
                                                        ↓
实际输入 x_eff → 原量化线性运算 → Add → 原 BN/激活/输出量化
       └────────────────→ 可训练低秩补偿分支 ─────────┘
```

对每个已经启用权重量化的 Conv2d/Linear：

```text
y = OriginalQuantizedLinear(x_eff) + Compensation(x_eff)
```

补偿必须加入原线性运算输出处，不能在已经经过 BN、ReLU 或输出量化的模块最终输出上再加一次。

训练过程中主分支继续执行原有 QAT fake quant；补偿分支与主分支共享同一个 `x_eff`，不重复执行激活量化；A/B 是可训练参数，随检测 loss 反向传播更新。原有 LSQ scale、SRQ 概率和 observer 按原训练策略工作。不加载训练完成模型后再做第二次 PTQ，不新增独立补偿 loss、教师模型、蒸馏目标或额外训练阶段。

默认补偿 rank 为 4，且不自动进行 rank sweep。

## 3. 补偿分支与 HGS 初始化

权重误差定义为：

\[
D=W_q-W_{ref}.
\]

普通卷积权重展平为 `[Cout, Cin*Kh*Kw]`，补偿因子转换为：

```text
B_kernel = B.T.reshape(rank, Cin, Kh, Kw)
A_kernel = A.reshape(Cout, rank, 1, 1)
```

B 使用原卷积的 kernel、stride、padding、dilation 和 padding mode；A 使用 1×1、stride=1、padding=0；补偿分支没有 bias、BN、ReLU 或额外 fake quant。

Linear 实现：

```text
Y = OriginalQuantizedLinear(X) + (X @ B) @ A.T
```

补偿模块插入 QAT 模型后，可使用当前权重量化状态和输入二阶矩做一次 HGS 初始化：

\[
H=\frac{1}{n}ZZ^T,\qquad
\widetilde H=H+\alpha\operatorname{mean}(\operatorname{diag}(H))I.
\]

对完整 `H_tilde` 执行 `torch.linalg.eigh`，按：

\[
s_i=\lambda_i\|Du_i\|_2^2
\]

选择固定 rank 个方向，并用：

\[
B=U[:,I],\qquad A=-DB
\]

初始化补偿参数。初始化完成后 A/B 不冻结，继续参与 QAT 训练。如果没有初始化统计数据，使用明确的 Xavier/零初始化 fallback；不能在训练结束后偷偷执行独立 Post-QAT 校准。

要求使用完整谱分解、未中心化二阶矩和 `D=Wq-Wref`，不使用 SVD、随机投影或只取最大特征值。不把 `Wq+AB^T` 合并后再次量化。

## 4. QAT 量化、梯度和计算边界

```text
原输入处理 / 原输入量化
          ↓
        x_eff
   ┌──────┴──────┐
   ↓             ↓
原量化 Conv/Linear   B → A
   └──────┬──────┘
          ↓ Add
          ↓
原 BN / ReLU / 输出量化
```

- 主分支继续使用原权重 fake quant、LSQ scale、STE 和 SRQ/QDrop 训练行为；
- A/B 是普通可训练参数，默认 FP32 保存和更新；
- 前向遵循现有 AMP/dtype 策略，不对整个模型调用 `.half()`；
- 补偿不额外执行激活量化；
- checkpoint 必须保存并恢复 A/B、rank、初始化状态和启用标记；
- 评估时直接使用训练完成 checkpoint，不再额外插入或初始化补偿模块。

目标层是实际启用权重量化的全部 Conv2d 和 Linear，包括 backbone、FPN、分类头和回归头。原配置明确保留浮点的首层、末层等例外不修改。共享模块只注册一次补偿模块，所有调用共享同一组 A/B。分组卷积如实际存在则按组独立统计和补偿，每组使用固定 rank。

## 5. 配置设计

新增集中配置：

```python
qat_compensation = dict(
    enabled=True,
    method='hgs_low_rank',
    rank=4,
    init='hgs',
    init_alpha=0.01,
    target='all_existing_weight_quantized_conv2d_and_linear',
    trainable=True,
    compensation_dtype='model',
)
```

`lsq` 与 `qdrop` 仍按现有配置互斥选择。补偿不是第三个 fake quant 方法，不应通过同时设置 `lsq=1` 和 `qdrop=1` 实现。

## 6. 代码改造范围

1. 新增可训练的 Conv2d、Linear 和必要的 grouped Conv 低秩补偿模块；
2. 在量化层的线性边界接入补偿，不使用错误的最终输出 hook；
3. 在 QAT 模型构建阶段按配置自动注入目标层；
4. 将现有 HGS solver 调整为补偿参数初始化器；
5. 确保 forward、反向传播、保存和恢复均能工作；
6. 将当前 Post-QAT `pipeline.py` 重构为初始化辅助代码，或删除不再需要的独立 plugin 流程；
7. 更新训练配置、训练脚本和使用文档；
8. `qat_compensation.enabled=False` 时保持原 QAT 行为不变。

## 7. 训练与评估

```text
1. 构建原 SSI + LSQ + SRQ QAT 模型。
2. 根据配置注入可训练补偿模块。
3. 使用 HGS 或明确 fallback 初始化 A/B。
4. 使用原 optimizer、loss 和训练入口继续 QAT。
5. 保存包含 A/B 的训练 checkpoint。
6. 使用同一 checkpoint 直接执行原评估流程。
```

本方案不要求训练完成后再次采集数据、构建独立插件或执行第二次 PTQ。正式实验比较原 QAT 与启用训练期低秩补偿的 QAT。

## 8. 验证要求

代码完成后只做配置解析、导入、语法和不加载数据的形状/state_dict 检查，不代跑训练、校准或数据集评估。检查包括目标层数量、A/B 形状和有限值、补偿输出形状、共享模块不重复注入以及 disabled 路径与原 QAT 一致。

## 9. 结论

本项目目标从“QAT 完成后再做 Post-QAT HGS/PTQ”改为：

> 在原有 SSI + LSQ + SRQ QAT 训练图中，加入 HGS 初始化的可训练低秩量化误差补偿分支，使补偿参数和检测模型一起学习。

这样只有一次 QAT 训练和一次最终评估，不把补偿误差校正实现成第二次量化后的独立 PTQ 流程。
