# Adaptive Low-bit Quantization for SAR Ship Detection
## 实验实现文档：SSI → Learned Step Size → SRQ

> 目的：把当前 ICASSP 2027 重构稿中的方法转换成可直接用于实验复现/代码实现的技术文档。
>
> 代码核对说明：本文以下实现映射以当前工程配置和 editable MQBench 源码为准。文中的方法名称与代码名称不完全相同：SSI 对应 `ClipStdObserver`，Task-Driven Learned Step Size 对应 `LearnableFakeQuantize`，SRQ 对应 `QDropFakeQuantize`。其中 `lsq` 和 `qdrop` 在现有配置中通过 `if/elif` 互斥；若要同时使用三者，需要采用 QDrop 分支并将 observer 改为 `ClipStdObserver`，同时补充 LSQ 的 gradient scaling。

---

# 1. 方法精简说明

## 1.1 任务定义

目标是在 SAR 舰船检测网络中进行 **联合权重—激活量化（joint weight–activation QAT）**。

例如：

- W8A8：8-bit weight + 8-bit activation；
- W6A6：6-bit weight + 6-bit activation；
- W4A4：4-bit weight + 4-bit activation。

本文针对 SAR 特征中少量高幅散射响应与大量中低幅有效响应并存的情况，将低比特量化问题概括为 **outlier-induced range–resolution conflict**：

- 量化范围过大：可以容纳极端响应，但量化步长增大，中低幅特征的表示分辨率降低；
- 量化范围过小：量化分辨率提高，但强散射响应更容易被 clipping。

因此，本文方法采用一个连续的三阶段过程：

\[
\boxed{\text{Initialization} \rightarrow \text{Adaptation} \rightarrow \text{Robustification}}
\]

对应：

\[
\boxed{\text{SSI} \rightarrow \text{Learned Step Size} \rightarrow \text{SRQ}}
\]

---

## 1.2 阶段一：Statistics-Guided Soft-Start Initialization（SSI，代码名 `ClipStdObserver`）

### 目的

在 QAT 开始前，为 activation quantizer 提供一个不易被极端散射值支配的初始 step size。

对于第 \(l\) 层 activation tensor \(X_l\)，计算：

\[
\mu_l = \mathrm{mean}(X_l),
\qquad
\sigma_l = \mathrm{std}(X_l).
\]

定义 statistics-derived initialization scale：

\[
R_l = \mu_l + k\sigma_l,
\]

再初始化量化步长：

\[
s_l^{(0)} =
\frac{R_l}{\sqrt{q_{\max,l}}}.
\]

其中 `std_scale` 控制分布尾部对初始化范围的影响；当前代码默认值为 `2.6`。

### 关键理解

这里的 \(R_l\) 应理解为 **用于初始化 step size 的统计尺度**，而不是严格意义上的最终 clipping endpoint。

SSI 只负责给 QAT 一个较合理的起点：

\[
s_l^{(0)}
\]

后续 step size 会继续根据 detection loss 学习。

### 代码对应关系

当前代码没有名为 `SSI` 的独立模块。其对应实现是 MQBench 的 `ClipStdObserver`：它计算输入的 mean/std，根据 `mean ± std_scale * std` 形成统计范围，再由 observer 的 `calculate_qparams()` 计算量化 scale 和 zero point。当前 observer 的默认 `std_scale=2.6`，并且统计范围还会与当前输入的真实 min/max 合并，不是严格只使用文档中的 `mu + k * sigma`。

配置可以同时对 weight 和 activation 使用该 observer；例如 LSQ 分支的 `w_observer` 和 `a_observer` 都设置为 `ClipStdObserver`。observer 在 fake-quantizer forward 时更新，不是文档伪代码中单独实现的全数据集 `sum/sq_sum/N` calibration collector。

---

## 1.3 阶段二：Task-Driven Learned Step Size

### 目的

SSI 只是初始化，最终量化步长由检测任务本身优化。

对标量 \(v\)，均匀 fake quantization 定义为：

\[
Q(v;s)
=
s\cdot
\mathrm{clip}
\left(
\left\lfloor
\frac{v}{s}
\right\rceil,
q_{\min},
q_{\max}
\right).
\]

对于 signed \(b\)-bit weight：

\[
q_{\min}=-2^{b-1},
\qquad
q_{\max}=2^{b-1}-1.
\]

对于 non-negative \(b\)-bit activation：

\[
q_{\min}=0,
\qquad
q_{\max}=2^b-1.
\]

由于 rounding 不可导，训练时采用 STE（Straight-Through Estimator）。

step size \(s\) 被设置为 learnable parameter，并与检测器参数一起通过 detection loss 优化：

\[
(\theta,s)
\leftarrow
(\theta,s)
-
\eta\nabla_{\theta,s}
\mathcal L_{\mathrm{det}}.
\]

为稳定 step-size 更新，采用 LSQ 风格的 gradient scaling：

\[
g_s=
\frac{1}{\sqrt{Nq_{\max}}},
\]

其中 \(N\) 为共享该 step size 的量化元素数。

### 代码对应关系

当前代码中的 Task-Driven Learned Step Size 由 MQBench 的 `LearnableFakeQuantize` 实现。它包含可学习 `scale`、STE rounding，以及 per-tensor/per-channel 的 LSQ gradient scaling；配置入口是 `w_fakequantize='LearnableFakeQuantize'` 和 `a_fakequantize='LearnableFakeQuantize'`。因此当前配置中的 `lsq=1` 已经对应本阶段。

需要区分 `QDropFakeQuantize`：它也包含可学习 scale 和 STE rounding，但其源码主要实现 QDrop 随机混合，并不自动等同于 `LearnableFakeQuantize` 的 LSQ gradient scaling。若要求 SSI + 严格 LSQ + SRQ 三者同时成立，需要在 QDrop fake quantizer 中补入 LSQ 的 gradient scaling，或实现一个组合 fake quantizer；仅把 `qdrop=1` 不能自动复用 `lsq=1` 分支。

### 这一阶段的作用

\[
\boxed{
\text{SSI provides initialization}
\quad\rightarrow\quad
\text{task loss corrects the initialization}
}
\]

因此，\(\mu+k\sigma\) 并不是固定的最终量化范围，而是一个 warm start。

---

## 1.4 阶段三：Stochastic Reconstruction Quantization（SRQ，代码名 `QDropFakeQuantize`）

### 目的

避免训练过程始终只暴露于一个固定的 deterministic quantization grid。

对于 activation \(X_l\)，先得到：

\[
X_l^q=Q(X_l;s_l).
\]

训练时采样 Bernoulli mask：

\[
M_l\sim \mathrm{Bernoulli}(p),
\]

并构造：

\[
\widetilde X_l
=
M_l\odot X_l^q
+
(1-M_l)\odot X_l.
\]

因此，一个 forward 中 activation 可以在：

\[
X_l
\leftrightarrow
X_l^q
\]

之间随机切换。

其中：

- \(p\)：选择 quantized activation 的概率；
- \(1-p\)：暂时保留 full-precision activation 的概率。

当前论文已有结果表明较好的设置约为：

\[
p=0.25.
\]

### 训练与推理不同

训练阶段：

\[
X_l \rightarrow \widetilde X_l.
\]

推理阶段关闭随机重构，全部走量化分支：

\[
X_l \rightarrow Q(X_l;s_l).
\]

所以 SRQ 只是 QAT 阶段的 regularization，不应给最终 inference 增加 FP branch。

### 代码对应关系

当前代码没有名为 `SRQ` 的独立模块。其对应实现是 MQBench 的 `QDropFakeQuantize`。该模块先执行 fake quantization，再使用 `torch.rand_like(X)` 和 `torch.where()` 在量化值与原始浮点值之间随机选择，因此 mask 粒度确定为 **element-wise**。

代码中的概率参数是 `quantizer.prob`：

- `prob=0.25`：约 25% 元素使用量化值；
- `prob=1.0`：全部使用量化值，不进行随机混合；
- 默认值为 `1.0`。

当前 QDrop 配置分支同时将 weight 和 activation fake quantizer 设置为 `QDropFakeQuantize`。如果只希望 activation 使用 SRQ，需要在模型初始化后只修改 activation quantizer 的 `prob`。

## 1.5 三者同时开启时的代码限制

当前配置文件使用互斥分支：

```python
if lsq:
    # LearnableFakeQuantize
elif qdrop:
    # QDropFakeQuantize
```

因此不能通过同时设置 `lsq=1` 和 `qdrop=1` 得到严格的三者组合。现有代码中有两种可区分的情况：

1. `lsq=1, qdrop=0`：`ClipStdObserver + LearnableFakeQuantize`，即 SSI-like 初始化 + 严格 LSQ；
2. `lsq=0, qdrop=1`：若将 qdrop 分支的 observer 改为 `ClipStdObserver`，再设置 `prob=0.25`，则是 SSI-like 初始化 + 可学习 QDrop scale + SRQ。

第二种情况中的 `QDropFakeQuantize` 虽然有 learnable scale 和 STE rounding，但不会自动继承 `LearnableFakeQuantize` 的 LSQ gradient scaling。若论文必须使用严格的 `SSI + LSQ + SRQ`，需要新增组合 fake quantizer，或在 QDrop 类中补入 LSQ gradient scaling；不能只依靠现有 config flags。

最终论文必须和实现保持一致。

---

# 2. 方法实现伪代码

## Algorithm 1：整体 QAT 流程

```text
Input:
    FP32 detector F(theta)
    training set D
    weight bit width bw
    activation bit width ba
    ClipStdObserver std_scale (default 2.6)
    QDropFakeQuantize probability p (set explicitly, e.g. 0.25)

Output:
    quantized detector Fq(theta, sw, sa)

-------------------------------------------------------
Stage 0: Build quantized detector
-------------------------------------------------------

1. Load / initialize FP32 detector F(theta).

2. For each quantizable convolution / linear layer l:
       attach weight quantizer Qw_l
       attach activation quantizer Qa_l

3. Set integer bounds:
       Weight:
           qmin_w = -2^(bw-1)
           qmax_w =  2^(bw-1)-1

       Activation (if non-negative):
           qmin_a = 0
           qmax_a = 2^ba - 1

4. Select one existing quantization branch:
       LSQ branch:
           w/a_observer = ClipStdObserver
           w/a_fakequantize = LearnableFakeQuantize
       or QDrop branch:
           w/a_observer = ClipStdObserver (if SSI-like initialization is wanted)
           w/a_fakequantize = QDropFakeQuantize
       The current config uses if/elif, so both branches cannot be
       selected simultaneously.

-------------------------------------------------------
Stage 1: Statistics-Guided Soft-Start Initialization
-------------------------------------------------------

5. Enable the observer and run the model forward.

6. For each fake quantizer using ClipStdObserver:
       compute mean/std of the current input;
       update observer min_val/max_val using std_scale;
       calculate scale and zero_point;
       copy them into the fake quantizer state.

7. With LearnableFakeQuantize, continue updating scale with
       LSQ gradient scaling during QAT.
   With QDropFakeQuantize, set quantizer.prob explicitly; its
       learnable scale does not automatically include LSQ gradient scaling.

-------------------------------------------------------
Stage 2 + Stage 3: QAT
-------------------------------------------------------

9. for epoch = 1 ... E:

10.     for (image, target) in D:

11.         for each quantized layer l:

12.             Quantize weight:
                   Wq_l = Q(W_l; sw_l)

13.             Compute full-precision pre-activation:
                   X_l = Layer(input; Wq_l)

14.             Quantize activation:
                   Xq_l = Q(X_l; sa_l)

15.             if training and SRQ enabled:
                   sample M_l ~ Bernoulli(p)
                   Xout_l = M_l * Xq_l + (1-M_l) * X_l
                else:
                   Xout_l = Xq_l

16.         Run detection head and obtain predictions.

17.         Compute detector loss:
                Ldet = Lcls + Lbox + ... (according to detector)

18.         Back-propagate with STE.

19.         In the LearnableFakeQuantize branch, use its built-in
            LSQ gradient scaling for learnable scale parameters.
            In the QDropFakeQuantize branch, use its implemented
            learnable scale and random mixing; add LSQ scaling only
            if the QDrop class has been extended accordingly.

20.         Update detector parameters theta
            and learnable step sizes sw, sa.

-------------------------------------------------------
Inference
-------------------------------------------------------

21. Disable SRQ stochastic mixing.

22. For every quantized layer:
        W -> Q(W; sw)
        X -> Q(X; sa)

23. Run fully quantized WbAb inference.
```

---

## Algorithm 2：SSI-like initialization (`ClipStdObserver`)

```text
Input:
    activation tensor stream {X_l}
    bit width b
    std_scale = 2.6     # current ClipStdObserver default

min_cur, max_cur = aminmax(X_l)
mu_l = mean(X_l)
sigma_l = std(X_l)

min_l = min(min_cur, mu_l - std_scale * sigma_l)
max_l = max(max_cur, mu_l + std_scale * sigma_l)

scale_l, zero_point_l = calculate_qparams(min_l, max_l)

return scale_l, zero_point_l
```

### 实验实现注意

当前 MQBench observer 不是文档原先建议的显式全局统计器。如果实验需要跨多个 batch 的严格总体统计，必须另行实现并记录；不能把 batch-level observer 更新描述为：

- running first moment；
- running second moment；

或者直接累计：

\[
\sum x,\qquad
\sum x^2,\qquad
N
\]

再计算：

\[
\mu=\frac{\sum x}{N},
\]

\[
\sigma=
\sqrt{
\frac{\sum x^2}{N}-\mu^2
}.
\]

这属于推荐实现方式，当前论文没有给出具体统计代码。

---

## Algorithm 3：SRQ

```text
Input:
    full-precision activation X
    quantized activation Xq
    quantized-path probability p

if training:
    M = Bernoulli(p) with the same broadcast structure
        required by the chosen SRQ granularity

    X_mix = M * Xq + (1 - M) * X
else:
    X_mix = Xq

return X_mix
```

---

# 3. 方法具体实现

# 3.1 总体代码结构

建议在现有 SAR detector 工程中新增如下结构：

```text
project/
├── models/
│   ├── detector/
│   └── quantization/
│       ├── quantizer.py
│       ├── ssi.py
│       ├── srq.py
│       ├── quant_conv.py
│       └── convert.py
├── tools/
│   ├── train_fp32.py
│   ├── train_qat.py
│   ├── eval_quant.py
│   └── collect_quant_stats.py
└── configs/
    ├── retinanet_r18_fp32.py
    ├── retinanet_r18_w8a8.py
    ├── retinanet_r18_w6a6.py
    └── retinanet_r18_w4a4.py
```

核心原则是：

> detector 本身尽量不改，量化逻辑封装为独立 quantizer 和 quantized layer wrapper。

这样方便做：

- FP32；
- LSQ only；
- SRQ only；
- SSI + LSQ；
- LSQ + SRQ；
- full method；

等消融。

---

# 3.2 基础 Fake Quantizer

实现一个统一量化器：

```python
class LSQFakeQuantizer(nn.Module):

    def __init__(self, bit, signed, learnable=True):
        super().__init__()

        self.bit = bit
        self.signed = signed

        if signed:
            self.qmin = -(2 ** (bit - 1))
            self.qmax =  (2 ** (bit - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = (2 ** bit) - 1

        self.step_size = nn.Parameter(torch.ones(1))
        self.initialized = False

    def forward(self, x):

        s = make_positive(self.step_size)

        x_int = round_ste(x / s)
        x_int = torch.clamp(
            x_int,
            self.qmin,
            self.qmax
        )

        x_q = x_int * s

        return x_q
```

其中 `round_ste()`：

```python
def round_ste(x):
    return (torch.round(x) - x).detach() + x
```

这样：

- forward 使用 `round(x)`；
- backward 对 rounding 近似为 identity。

---

# 3.3 Step-size Gradient Scaling

LSQ 风格的 step-size gradient scale：

\[
g=\frac{1}{\sqrt{Nq_{\max}}}.
\]

可通过 gradient-scale trick 实现：

```python
def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad
```

量化时：

```python
g = 1.0 / math.sqrt(numel_shared * qmax)
s = grad_scale(step_size, g)
```

然后再进行：

```python
x_int = round_ste(x / s)
x_q = clamp(x_int) * s
```

这样可以避免 step size 的梯度尺度远大于网络参数。

---

# 3.4 SSI 统计收集器

建议 activation quantizer 增加：

```python
observer_enabled = True
ssi_initialized = False
```

统计模式下：

```python
class SSIObserver:

    def __init__(self):
        self.sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, x):
        self.sum += x.sum()
        self.sq_sum += (x * x).sum()
        self.count += x.numel()

    @torch.no_grad()
    def get_mean_std(self):
        mu = self.sum / self.count
        var = self.sq_sum / self.count - mu * mu
        sigma = torch.sqrt(torch.clamp(var, min=0))
        return mu, sigma
```

然后：

```python
@torch.no_grad()
def ssi_initialize(quantizer, observer, k):

    mu, sigma = observer.get_mean_std()

    R = mu + k * sigma

    s0 = R / math.sqrt(quantizer.qmax)

    quantizer.step_size.copy_(s0)

    quantizer.initialized = True
```

---

# 3.5 SSI 的推荐实验流程

### Step A：载入 FP32 检测器

```text
FP32 RetinaNet-R18
      ↓
load pretrained / trained SAR checkpoint
```

### Step B：替换 quantizable layers

例如：

```text
Conv2d
   ↓
QuantConv2d
   ├── weight quantizer
   └── activation quantizer
```

### Step C：由 observer 收集统计并初始化量化参数

推荐：

```python
model.eval()

with torch.no_grad():
    for i, data in enumerate(calibration_loader):

        model(data)

        if i == num_calibration_batches - 1:
            break
```

这里不调用文档中虚构的 `initialize_ssi()` 或 `observer.get_mean_std()`。`ClipStdObserver` 在 fake quantizer 的 forward 中计算当前输入的 mean/std、更新 `min_val`/`max_val`，再由 `calculate_qparams()` 初始化 `LearnableFakeQuantize.scale`。observer 的开启/关闭由量化框架的 `observer_enabled` 控制。

当前实现默认 `ClipStdObserver.std_scale=2.6`；如果需要单独的 calibration 阶段，必须使用真实训练脚本和 MQBench 的 observer 状态控制，并记录实际 batch 数量，不能套用本文原先的 `k=3.0` 和 `R/sqrt(qmax)` 伪代码。

---

# 3.6 权重量化实现

对于卷积层：

```python
class QuantConv2d(nn.Conv2d):

    def __init__(...):
        ...
        self.weight_quant = LSQFakeQuantizer(
            bit=w_bits,
            signed=True
        )

        self.act_quant = LSQFakeQuantizer(
            bit=a_bits,
            signed=False
        )

    def forward(self, x):

        wq = self.weight_quant(self.weight)

        y_fp = F.conv2d(
            x,
            wq,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

        ...
```

注意这里：

- 权重实际 forward 使用 \(W_q\)；
- 反向仍更新 FP parameter \(W\)；
- 下一次 forward 再重新 fake quantize。

即典型 QAT：

\[
W
\rightarrow
Q(W)
\rightarrow
\text{forward}
\]

但 optimizer 更新的是：

\[
W.
\]

---

# 3.7 Activation + SRQ（`QDropFakeQuantize`）实现

在卷积/激活输出后：

```python
x_fp = y_fp

x_q = self.act_quant(x_fp)

if self.prob < 1.0:
    # torch.rand_like gives an element-wise mask.
    x_out = torch.where(
        torch.rand_like(x_q) < self.prob,
        x_q,
        x_fp
    )
else:
    x_out = x_q

return x_out
```

这里的 `prob` 是量化路径概率，不是 dropout 概率。`QDropFakeQuantize` 默认 `prob=1.0`；训练时需要显式设置，例如：

```python
for module in model.modules():
    if module.__class__.__name__ == 'QDropFakeQuantize':
        module.prob = 0.25
```

---

# 3.8 SRQ 在 inference 中必须关闭

训练：

```python
model.train()
for module in model.modules():
    if module.__class__.__name__ == 'QDropFakeQuantize':
        module.prob = 0.25
```

推理：

```python
model.eval()
for module in model.modules():
    if module.__class__.__name__ == 'QDropFakeQuantize':
        module.prob = 1.0
```

最终：

```python
x_out = x_q
```

因此最终 W4A4 推理应保持：

\[
W\rightarrow W_4,
\qquad
A\rightarrow A_4.
\]

不能在测试阶段继续混入 FP activation，否则测试结果不再是真正的 W4A4。

---

# 3.9 整体 QAT Training Loop

推荐实现为：

```python
model = build_fp32_detector()
load_fp32_checkpoint(model)

model = convert_to_quantized_model(
    model,
    w_bits=4,
    a_bits=4
)

# --------------------------------------
# QAT configuration
# --------------------------------------
# Use ClipStdObserver + QDropFakeQuantize in the qdrop branch.
# The observer initializes scale during fake-quantizer forward.
enable_observer(model)
enable_quantization(model)

for module in model.modules():
    if module.__class__.__name__ == 'QDropFakeQuantize':
        module.prob = 0.25

optimizer = build_optimizer(model)

for epoch in range(num_epochs):

    model.train()

    for images, targets in train_loader:

        optimizer.zero_grad()

        outputs = model(images, targets)

        loss = sum(outputs.values())

        loss.backward()

        optimizer.step()

    # Before validation, set QDropFakeQuantize.prob = 1.0.
    evaluate(model, val_loader)

# --------------------------------------
# Final evaluation
# --------------------------------------

model.eval()
for module in model.modules():
    if module.__class__.__name__ == 'QDropFakeQuantize':
        module.prob = 1.0

evaluate(model, test_loader)
```

---

# 3.10 RetinaNet 中建议的插入位置

当前论文使用：

- RetinaNet；
- ResNet-18 backbone；
- FPN channel width = 64。

建议对主要 Conv 层执行 weight fake quantization，并对量化层输出 activation 执行 activation fake quantization。

概念上：

```text
Input
  ↓
Backbone Conv
  ↓
W Quant
  ↓
Conv
  ↓
Activation Quant
  ↓
SRQ Mix
  ↓
Next Layer
```

FPN、classification head、regression head 同样按配置插入。

但以下内容当前论文没有明确：

- stem / first convolution 是否量化；
- RetinaNet 最后 prediction layers 是否保持 FP32；
- BatchNorm 是否 fold；
- ReLU 前还是 ReLU 后做 activation quantization；
- FPN lateral/output conv 是否全部量化。

这些必须从真实代码确认。

---

# 3.11 当前代码的配置接口

当前工程使用 `if/elif` 选择量化方法，不能同时把 `lsq` 和 `qdrop` 设为 1。严格按现有代码，LSQ 配置如下：

```python
bit = 4
lsq = 1
qdrop = 0

# lsq branch: SSI-like observer + strict LSQ learnable fake quantization
'w_observer': 'ClipStdObserver'
'a_observer': 'ClipStdObserver'
'w_fakequantize': 'LearnableFakeQuantize'
'a_fakequantize': 'LearnableFakeQuantize'
```

若启用代码中对应 SRQ 的 QDrop，则使用：

```python
bit = 4
lsq = 0
qdrop = 1

# 为了保留 SSI-like 初始化，需要将 qdrop 分支中的 observer 从
# EMAMinMaxObserver 改为 ClipStdObserver。
'w_observer': 'ClipStdObserver'
'a_observer': 'ClipStdObserver'
'w_fakequantize': 'QDropFakeQuantize'
'a_fakequantize': 'QDropFakeQuantize'
```

随后在模型 prepare 完成后设置：

```python
for module in model.modules():
    if module.__class__.__name__ == 'QDropFakeQuantize':
        module.prob = 0.25
```

这个配置实现 SSI-like initialization + learnable QDrop scale + SRQ，但 QDrop 本身没有自动复用 `LearnableFakeQuantize` 的 LSQ gradient scaling。要严格实现 SSI + LSQ + SRQ，必须新增组合 fake quantizer或修改 MQBench 的 QDrop 类。

W8A8/W6A6/W4A4 只修改：

```python
w_bits
a_bits
```

即可。

---

# 3.12 消融实验如何通过配置直接实现

## B0：基础 fixed-step QAT

```python
fixed = 1
lsq = 0
qdrop = 0
```

需要明确定义 fixed step 的初始化方式。

---

## B1：SSI only

```python
fixed = 1
lsq = 0
qdrop = 0
# 使用 ClipStdObserver，但关闭后续 learnable scale 更新。
```

即：

- SSI 初始化 step；
- 之后 step 固定。

---

## B2：LSQ only

```python
fixed = 0
lsq = 1
qdrop = 0
```

当前已有 SSDD W4A4：

\[
\mathrm{mAP}_{50}=90.7.
\]

---

## B3：SRQ only

```python
fixed = 0
lsq = 0
qdrop = 1
```

当前论文文字已有 SSDD W4A4：

\[
\mathrm{mAP}_{50}=89.7.
\]

但必须明确定义其固定 quantizer initialization。

---

## B4：SSI + LSQ

```python
fixed = 0
lsq = 1
qdrop = 0
# lsq 分支的 observer 使用 ClipStdObserver。
```

结果待补。

---

## B5：SSI-like + QDrop/SRQ

```python
fixed = 0
lsq = 0
qdrop = 1
# qdrop 分支的 observer 改为 ClipStdObserver，并设置 prob=0.25。
```

结果待补。

---

## B6：QDrop/SRQ with learnable scale

```python
fixed = 0
lsq = 0
qdrop = 1
# 当前代码没有同时继承 LearnableFakeQuantize 的 LSQ gradient scaling。
```

结果待补。

---

## B7：SSI-like + learnable QDrop + SRQ

```python
fixed = 0
lsq = 0
qdrop = 1
# qdrop observer 改为 ClipStdObserver，prob=0.25；严格 LSQ scaling
# 需要组合 fake quantizer，不能仅通过现有 flags 打开。
```

当前 SSDD W4A4：

\[
\mathrm{mAP}_{50}=92.5.
\]

---

# 3.13 当前论文对应的基础实验配置

目前论文材料已经支持的设置：

### Detector

```text
RetinaNet
Backbone: ResNet-18
FPN width: 64
```

### Image preprocessing

```text
Long side: 1333
Short side: 800
Random horizontal flip: p = 0.5
Minimum valid target size: 32 pixels
```

### Training

```text
Epochs: 300
GPU: NVIDIA GeForce RTX 2080 Ti
```

### Datasets

#### SSDD

```text
1160 images
2456 ship instances
train:test = 8:2
```

#### AIR-SARShip-1.0

```text
crop size = 512 × 512
overlap = 50%
812 ship-containing patches
train:val:test = 8:1:1
```

#### HRSID

```text
5604 cropped images
16951 ship instances
official split = 65% / 35%
```

### Quantization

```text
W8A8
W6A6
W4A4
```

### 当前已有关键超参数叙事

```text
SSI-like observer: `ClipStdObserver`, default `std_scale = 2.6`
SRQ/QDrop: `QDropFakeQuantize.prob = 0.25` during training
```

---

# 3.14 仍然必须从代码/实验日志确认的实现项

以下配置是最终复现实验文档中不能缺的，但当前论文材料没有提供：

1. optimizer；
2. initial learning rate；
3. LR schedule；
4. batch size；
5. weight decay；
6. warm-up；
7. QAT 是否从 FP32 pretrained checkpoint 开始；
8. FP32 checkpoint 的训练方式；
9. weight quantization 是 per-tensor 还是 per-channel；
10. activation quantization 是 per-tensor 还是 per-channel；
11. activation 是否全部 non-negative；
12. first layer 是否量化；
13. last prediction layer 是否量化；
14. BatchNorm 的处理方式；
15. 实际 observer 运行了多少个 training/calibration batch；
16. `ClipStdObserver` 是否在最终实验中同时用于 weight 和 activation；当前 LSQ 配置是二者都使用；
17. QDrop 的 mask granularity 已确认是 element-wise；
18. `QDropFakeQuantize.prob` 是否作用于所有 fake quantizers，还是仅指定 activation quantizers；
19. validation 时是否显式将 QDrop 的 `prob` 恢复为 `1.0`；
20. 是否使用 AMP；
21. 随机种子及重复实验次数。

这些应视为 **P0 实现核对项**。

---

# 4. 一次完整实验应该怎样执行

推荐整个实验严格按下面顺序运行。

## Step 1：FP32 baseline

训练/加载：

```text
RetinaNet-R18-FPN64 FP32
```

得到三个数据集的：

\[
\mathrm{mAP}_{50}^{FP32}.
\]

---

## Step 2：LSQ baseline

分别运行：

```text
W8A8
W6A6
W4A4
```

不启用 SSI，不启用 SRQ。

目的是得到最关键 baseline：

\[
\mathrm{LSQ}.
\]

---

## Step 3：Full method

分别运行：

```text
SSI + LSQ + SRQ
```

在：

```text
W8A8 / W6A6 / W4A4
```

下测试。

---

## Step 4：完整消融

固定：

```text
Dataset = SSDD
Detector = RetinaNet-R18
Bit width = W4A4
```

运行 8 个组合：

```text
Base
SSI
LSQ
SRQ
SSI+LSQ
SSI+SRQ
LSQ+SRQ
SSI+LSQ+SRQ
```

---

## Step 5：超参数实验

固定 full framework，仅修改：

\[
k
\]

和：

\[
p.
\]

如果之前没有真实 sweep，可新增：

```text
k ∈ {1, 2, 3, 4, 5}
p ∈ {0, 0.125, 0.25, 0.5, 0.75}
```

但这组 grid 是推荐新增实验，不是当前论文已有实验事实。

---

## Step 6：SAR-specific diagnostic

对若干代表层保存 FP activation：

```text
early backbone
late backbone / FPN
detection head
```

统计：

\[
\max |X|,
\]

\[
P_{99}(|X|),
\]

\[
P_{99.9}(|X|),
\]

以及：

\[
\rho =
\frac{\max|X|}
{P_{99}(|X|)+\epsilon}.
\]

再测试不同初始化 range / \(k\) 下：

- clipping error；
- rounding error；
- total quantization error。

用于验证：

\[
\boxed{
\text{outlier-induced range--resolution conflict}
}
\]

而不是只在 Introduction 中口头声称。

---

## Step 7：Cross-detector

时间有限时只增加一个：

```text
FCOS
```

或：

```text
Faster R-CNN
```

固定：

```text
Dataset = SSDD
Bit = W4A4
```

比较：

```text
FP32
LSQ
Ours
```

即可验证方法是否只对 RetinaNet 有效。

---

# 5. 最终方法实现逻辑总结

整个方法可以压缩成：

```text
FP32 SAR detector
        │
        ▼
Insert W/A fake quantizers
        │
        ▼
Collect activation statistics
        │
        ▼
SSI:
mu, sigma → initial activation step size
        │
        ▼
QAT:
learn W, detector parameters, and quantization step sizes
        │
        ▼
SRQ during training:
random FP / quantized activation exposure
        │
        ▼
Disable SRQ
        │
        ▼
Fully quantized WbAb inference
```

其核心不是三个互不相关的模块，而是：

\[
\boxed{
\text{better initialization}
\rightarrow
\text{task-adaptive quantization}
\rightarrow
\text{robust low-bit training}
}
\]

即：

\[
\boxed{
\text{SSI}
\rightarrow
\text{Learned Step Size}
\rightarrow
\text{SRQ}.
}
\]
