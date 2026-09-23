# 可直接发送给 Codex 的实施 Prompt：QAT 训练期量化误差补偿

## 1. 任务目标

当前项目已经有 SSI + LSQ + SRQ 的 QAT 实现。不要再实现“训练完成后加载 checkpoint、重新校准、再做一次 PTQ/HGS”的 Post-QAT 流程。该方向不作为本轮主线。

本轮要实现的是：**在 QAT 模型构建和训练过程中加入 HGS 初始化的可训练低秩量化误差补偿分支。** 补偿参数 A/B 和原检测模型一起参与 QAT 训练，训练结束后直接使用同一个 checkpoint 评估。

HGS 仅作为低秩补偿的初始化原理和方向选择方法，不作为第二次量化或训练后的独立插件。

## 2. 执行边界

- 不修改现有 `ClipStdObserver`、`LearnableFakeQuantize`、`QDropFakeQuantize` 的核心实现。
- 不把 `lsq=1` 和 `qdrop=1` 同时打开；现有两个 fake quant 分支仍按原配置选择。
- 不重新实现 SSI、LSQ、SRQ。
- 不添加 Post-QAT PTQ、独立 plugin eval 或第二次量化路径。
- 不添加新的 teacher、蒸馏 loss、独立 reconstruction loss、门控系数或额外 optimizer。
- 不改变原有 QAT 默认行为；`qat_compensation.enabled=False` 时应与当前流程一致。
- 目标层覆盖当前模型中实际启用权重量化的全部 Conv2d 和 Linear；原精度例外保持不变。
- 默认固定 `rank=4`，不做 rank sweep、自适应 rank 或层筛选。
- 先完成代码和配置，再只做导入、语法、配置解析和不加载数据的形状检查；不要替用户启动训练、校准或评估。

## 3. 需要先确认的代码事实

阅读当前仓库的 QAT 构建、训练和评估入口，确认：

1. 量化 Conv/Linear 的实际类型和 forward 边界；
2. `Wref` 在哪里保存，原权重量化器如何得到反量化 `Wq`；
3. 输入 fake quant 和输出 fake quant 的准确位置；
4. BN、ReLU、Conv-BN 融合和共享模块的情况；
5. SRQ/QDrop 的真实关闭方式；
6. checkpoint 的保存/恢复路径；
7. 现有 QAT 配置如何增加自定义模块。

不能用普通 `nn.Conv2d` 代替实际量化模块，也不能猜测不存在的属性名。若实际模块无法在其线性边界插入补偿，应完成必要的模块适配，而不是使用错误的最终输出 hook。

## 4. 目标计算图

对目标层，必须实现：

```text
原输入处理 / 输入量化
          ↓
        x_eff
   ┌──────┴──────┐
   ↓             ↓
原量化线性运算   B → A 低秩补偿
   └──────┬──────┘
          ↓ Add
          ↓
原 BN / ReLU / 输出量化 / 后续处理
```

主分支和补偿分支使用同一个 `x_eff`。补偿不额外执行激活量化，不绕过主分支输入处理。不能无条件写成 `original_block(x) + low_rank(x)`，除非已验证模块只包含目标线性运算。

## 5. 补偿模块

新增或改造模块，使 A/B 是可训练参数而不是冻结 buffer。

### Conv2d

普通卷积的权重矩阵为 `[Cout, Cin*Kh*Kw]`。因子转换为：

```text
B_kernel = B.T.reshape(rank, Cin, Kh, Kw)
A_kernel = A.reshape(Cout, rank, 1, 1)
```

B 使用原卷积的 kernel、stride、padding、dilation 和 padding mode；A 使用 1×1、stride=1、padding=0；不带 bias、BN、ReLU 或额外 fake quant。

### Linear

实现：

```text
Y = OriginalQuantizedLinear(X) + (X @ B) @ A.T
```

保留输入前导维度和原输出形状。

### Grouped Conv

如果当前模型存在量化 grouped Conv，则每组独立统计、独立初始化和独立补偿；组间不新增连接。每组默认 rank=4，仅当组输入维度小于 4 时使用 `min(4, d_group)`，并记录有效 rank。

## 6. HGS 初始化

补偿模块插入后，在训练开始前或训练初期执行一次初始化。初始化使用当前模型的权重量化状态和输入二阶矩，不使用检测 loss 反向传播。

```python
D = Wq.float() - Wref.float()
H = second_moment_sum / n_patches
H = 0.5 * (H + H.T)
H_tilde = H + alpha * H.diagonal().mean() * I
lambda_, U = torch.linalg.eigh(H_tilde)
P = D @ U
scores = lambda_ * P.square().sum(dim=0)
idx = stable_descending_argsort(scores)[:min(rank, d_in)]
B = U[:, idx]
A = -(D @ B)
```

要求：

- 使用完整谱分解后再按贡献分数选方向；
- 不使用 SVD、只取最大特征值或随机投影；
- 统计使用未中心化二阶矩；
- `D = Wq - Wref`，符号不能反；
- HGS 计算使用 FP32；
- 初始化后的 A/B 转为模型训练参数并允许梯度更新；
- HGS 初始化不得在每个 iteration 重复执行；
- 没有校准输入时，提供明确的 Xavier/零初始化 fallback，并记录初始化方式；
- 不把 `Wq + A @ B.T` 合并后再次量化。

初始化使用的输入统计可以来自训练开始前固定的小批训练样本，但不能形成独立的 Post-QAT 运行入口，也不能在 QAT 训练结束后重新执行。

## 7. QAT 梯度和量化状态

- 主分支继续使用原权重 fake quant、LSQ scale 和 STE；
- SRQ/QDrop 继续按现有 QAT 训练策略工作；
- A/B 是普通 `nn.Parameter`，默认 FP32 保存和更新；
- 前向遵循现有 AMP/dtype 策略，不对整个模型调用 `.half()`；
- 补偿分支不增加新的 fake quant，除非代码事实证明原线性边界必须共享已有输出量化；
- observer、BN 和量化步长沿用现有训练配置；
- 保存 checkpoint 时 A/B、rank、enabled、初始化标志和版本信息必须进入 `state_dict` 或等价 checkpoint 状态；
- 加载 checkpoint 时恢复同一补偿结构，不能把普通旧 checkpoint 冒充成已经训练过补偿的模型。

## 8. 配置和入口

新增集中配置，形式可以适配当前仓库：

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

要求：

1. QAT 构建阶段根据 `enabled` 注入补偿模块；
2. 训练入口继续使用原 optimizer 和原 loss；
3. 训练结束后原评估入口直接读取包含 A/B 的 checkpoint；
4. 禁用配置时不注入任何补偿模块；
5. 不再以 `qat_checkpoint + hgs_plugin.pt` 作为正式运行前提；
6. 可以保留数学 solver 作为初始化工具，但删除/废弃训练完成后独立构建 plugin 的主入口。

## 9. 建议代码组织

```text
quantization/hgs/
  solver.py          # HGS 完整谱分解和初始化因子
  moments.py         # 训练开始前输入二阶矩统计
  compensation.py    # 可训练 Conv/Linear 低秩补偿模块
  adapter.py         # 实际量化层边界和 Wref/Wq 适配
  injection.py       # QAT 构建阶段注入/恢复补偿
configs/              # 启用补偿的 QAT 配置
tools/                # 仅保留训练/评估所需入口适配
docs/                 # 更新训练期补偿说明
```

当前 `residual.py` 如果仍使用 FP16 buffer，应改造成训练期可训练参数模块；当前 `pipeline.py` 如果只服务于 Post-QAT 插件生成，应重构为初始化辅助代码或移除，不得继续作为主流程。

## 10. 必须保留的检查

- 目标层数量和权重量化状态检查；
- A/B 的形状、rank、有限值检查；
- 补偿输出与主线性输出形状检查；
- shared module 不重复注入；
- checkpoint 中补偿状态和配置一致性检查；
- disabled 路径与原 QAT 行为一致。

不添加 mAP 必须提升、量化误差必须为零或 headroom 必须为正等运行门槛。

## 11. 最终交付

最终说明：

1. 修改了哪些量化层和线性边界；
2. A/B 如何初始化、如何参与梯度训练；
3. SSI、LSQ、SRQ 如何保持原逻辑；
4. 哪些 QAT 配置启用了补偿；
5. checkpoint 如何保存和恢复；
6. 做了哪些静态检查；
7. 明确尚未代跑训练和数据集评估。

本轮的最终目标是一次完整的“带训练期量化误差补偿的 QAT”，而不是“QAT 之后再做一次 PTQ”。
