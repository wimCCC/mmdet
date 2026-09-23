# SSDD RetinaNet-R50 量化实验 Recipe

## 1. 已确认的问题

之前 W4A4/W6A6 实验出现 mAP=0，主要不是 SSDD 标注或 COCO evaluator 故障，而是基础训练 recipe 有问题：

1. **学习率过低**：旧配置使用 `lr=1e-5`，而可正常训练的 RetinaNet 配置使用 `lr=0.003`，相差约 300 倍。
2. **缺少 warmup**：旧 QAT 配置没有经过 500 iteration 的 LinearLR warmup。
3. **分类损失不正确**：旧配置使用了 `CrossEntropyLoss(use_sigmoid=True)`；RetinaNet 应使用 FocalLoss 来处理大量负 anchor。
4. **量化和训练问题同时混合**：旧实验同时使用了错误的 FP32 优化 recipe，因此不能据此得出 W4A4/W6A6 不可训练的结论。
5. **实验变量过多**：正式比较前不能同时加入 SRQ、HGS compensation 和 FP32 head 豁免，否则无法定位性能变化来源。

SSDD 标注已检查：928 张图像、2041 个标注、单类别 `ship`、无空 bbox。

## 2. 已验证的 FP32 基线

配置：

- `configs/_hgs_retina/ssdd_retinanet_r50_fp32_30e_lr3e-3.py`
- RetinaNet-R50
- FocalLoss：`gamma=2.0`、`alpha=0.25`、`loss_weight=1.0`
- SGD：`lr=0.003`、`momentum=0.9`、`weight_decay=1e-4`
- LinearLR warmup：500 iterations，`start_factor=0.001`
- MultiStepLR：epoch `[20, 27]`，`gamma=0.1`
- 30 epochs
- 不使用 `qat_pretrained`
- 使用 GPU 2，不使用 GPU 7

最终验证结果：

| Epoch | mAP | mAP50 | mAP75 | AR |
|---:|---:|---:|---:|---:|
| 30 | 0.262 | 0.680 | 0.174 | 0.370 |

最佳权重：

`work_dirs/hgs_retina/ssdd_retinanet_r50_fp32_30e_lr3e-3/best_coco_bbox_mAP_50_epoch_30.pth`

## 3. 当前 W6A6 基线

配置：

- `configs/_hgs_retina/ssdd_retinanet_r50_w6a6_ssi_lsq_30e_lr3e-3.py`
- W6A6
- SSI：`ClipStdObserver`
- LSQ：`LearnableFakeQuantize`
- 权重对称量化，激活非对称量化
- FocalLoss
- 与 FP32 基线相同的 SGD、warmup 和 scheduler
- `qat_compensation.enabled=False`
- 不使用 SRQ
- 不使用 `qat_pretrained`
- 不使用 GPU 7

启动命令：

```bash
PYTHONPATH=/data5/caiwm/mmdet \
CUDA_VISIBLE_DEVICES=2 \
/data5/caiwm/miniforge3/envs/tet/bin/python \
tools/train.py \
configs/_hgs_retina/ssdd_retinanet_r50_w6a6_ssi_lsq_30e_lr3e-3.py
```

## 4. 后续实验顺序

必须保持基础 recipe 不变，只逐步改变量化因素：

1. 完成 W6A6 SSI+LSQ 30 epoch，作为第一条 QAT 基线。
2. 使用完全相同的训练配置运行 W4A4 SSI+LSQ 30 epoch。
3. 如果全量化 head 导致性能明显下降，再增加 `bbox_head` FP32 ablation：
   - `qat_fp32_weight_patterns = [r'^bbox_head']`
   - `qat_fp32_activation_patterns = [r'^bbox_head']`
4. 在确认纯 SSI+LSQ 的结果后，再单独加入 HGS compensation。
5. SRQ 不作为当前主线；如需对比，必须单独建立同一训练 recipe 下的 SRQ 实验。

## 5. 训练约束

所有训练命令都应包含仓库根目录：

```bash
PYTHONPATH=/data5/caiwm/mmdet
```

这样才能解析 `mmdet/models/dense_heads/anchor_head.py` 中的 `vp_qod` 导入。

不要在实验中自动添加 `qat_pretrained`。除非配置明确指定，否则模型使用自身初始化；当前 recipe 默认加载 torchvision ResNet-50 backbone 初始化。

## 6. 结论

旧的 mAP=0 结果应标记为 **invalid diagnostic runs**，不能作为量化能力结论。后续比较必须以已验证的 FP32 recipe 为共同起点：

`FocalLoss + SGD(lr=0.003) + 500-iter warmup + 相同数据/模型/评估器`。

## 7. 已清理的旧 Ablation Workdir

以下目录属于旧实验产物，包含 SRQ、旧版 LSQ、compensation、HGS 或不同 seed/rank 的历史 ablation。它们不属于当前有效的 R50 recipe，已从 `work_dirs/` 删除；实验结论不再依赖这些旧权重：

- `ssdd_retina_ablation_lsq`
- `ssdd_retina_ablation_lsq_comp`
- `ssdd_retina_ablation_lsq_comp_hgs`
- `ssdd_retina_ablation_lsq_comp_hgs_seed2026`
- `ssdd_retina_ablation_lsq_comp_rank8`
- `ssdd_retina_ablation_lsq_comp_seed2026`
- `ssdd_retina_ablation_lsq_srq`
- `ssdd_retina_ablation_lsq_srq_anneal_comp`
- `ssdd_retina_ablation_lsq_srq_comp`
- `ssdd_retina_r18_ssi_lsq_srq_hgs_ablation`

保留的实验只包括当前验证过的 FP32 基线和正在进行的 W6A6 SSI+LSQ 基线。后续新实验必须使用新的独立 workdir，并在本文件中记录配置、训练 recipe 和指标。

## 8. Ablation 消融分析结论

### 8.1 当前可以确认的结论

| 消融因素 | 当前结论 | 依据/状态 |
|---|---|---|
| FP32 训练 recipe | 必须先修正，否则量化结果没有解释价值 | `lr=1e-5` 的旧配置无有效学习；修正为 `lr=0.003`、FocalLoss 和 warmup 后，30 epoch 达到 mAP50=`0.680` |
| FocalLoss vs. 旧 BCE/CrossEntropy 配置 | RetinaNet 应使用 FocalLoss | 旧 `CrossEntropyLoss(use_sigmoid=True)` 会使分类学习异常；FocalLoss 后 FP32 正常收敛 |
| SSI + LSQ | 当前主线 | 使用 `ClipStdObserver` + `LearnableFakeQuantize`，先单独验证，不混入其他补偿方法 |
| W6A6 vs. W4A4 | 尚无可比结论 | 旧 W6A6/W4A4 使用了错误的 `lr=1e-5` recipe；必须用统一的新 recipe 重跑 |
| HGS compensation | 尚无可比结论 | 旧实验与其他变量混合；应在纯 SSI+LSQ 基线之后单独加入 |
| SRQ | 暂不作为主线 | 当前研究约束是不使用 SRQ；若以后比较，必须使用与 SSI+LSQ 完全相同的训练 recipe |
| FP32 bbox head | 尚无可比结论 | 只有在全量化 head 明显掉点后，才运行 head-FP32 ablation |

### 8.2 旧 ablation 结果的处理

已删除的旧 ablation workdir 没有保留可复核的完整指标表，因此这里**不虚构旧实验的数值 mAP 结论**。这些结果统一标记为 `invalid / not comparable`，原因是它们可能混用了旧学习率、缺失 warmup、SRQ、HGS compensation、不同 seed 或不同 rank。

因此当前唯一可作为定量参照的结果是：

- FP32 R50：mAP=`0.262`，mAP50=`0.680`，mAP75=`0.174`，AR=`0.370`；
- W6A6 SSI+LSQ：已完成统一 recipe 训练，30 epoch 结果为 mAP=`0.064`、mAP50=`0.226`；
- W4A4 SSI+LSQ、head-FP32、HGS：尚未在统一 recipe 下完成。

### 8.4 W6A6 SSI+LSQ 统一 Recipe 结果

当前 W6A6 训练已经完成 30 epoch，配置与 FP32 基线保持一致：

| 配置 | mAP | mAP50 | mAP75 | AR |
|---|---:|---:|---:|---:|
| FP32 R50，30 epoch | 0.262 | 0.680 | 0.174 | 0.370 |
| W6A6 SSI+LSQ，30 epoch | 0.064 | 0.226 | 0.008 | 0.183 |

W6A6 的最佳 checkpoint 为：

`work_dirs/hgs_retina/ssdd_retinanet_r50_w6a6_ssi_lsq_30e_lr3e-3/best_coco_bbox_mAP_50_epoch_30.pth`

该结果说明：

1. 修正学习率、FocalLoss 和 warmup 后，W6A6 不再是完全不学习；mAP50 从 epoch 5 的 `0` 上升到 epoch 30 的 `0.226`。
2. 但全模型 W6A6 仍明显低于 FP32：mAP50 下降 `0.454`，相对 FP32 约下降 `66.8%`。
3. 这表明旧的 mAP=0 确实部分来自错误 recipe，但当前主要瓶颈已经转移到量化本身，尤其是完整量化的 RetinaHead/FPN 激活和 LSQ scale 学习。
4. 现在不应直接加入 HGS 或 SRQ。下一步应先运行同一 recipe 的 `bbox_head` FP32 ablation，判断检测 head 量化是否是主要损失来源；之后再考虑 W4A4 和 HGS。

### 8.5 W6A6 200 Epoch Run 状态

200 epoch 长跑已按用户要求手动停止。该运行使用 GPU 6，未使用 GPU 7；停止时训练进行到约 epoch 99，因此不是完整 200 epoch 结果。

已完成的验证结果：

| Epoch | mAP | mAP50 | mAP75 | AR |
|---:|---:|---:|---:|---:|
| 20 | 0.131 | 0.468 | 0.017 | 0.228 |
| 30 | 0.180 | 0.577 | 0.045 | 0.260 |
| 40 | 0.186 | 0.596 | 0.050 | 0.257 |
| 50 | 0.192 | 0.613 | 0.045 | 0.268 |
| 60 | 0.192 | 0.611 | 0.053 | 0.264 |
| 70 | 0.198 | 0.593 | 0.056 | 0.282 |
| 80 | 0.175 | 0.575 | 0.035 | 0.254 |
| 90 | 0.209 | 0.616 | 0.065 | 0.274 |

停止前最近一次已保存 checkpoint 为 epoch 90，路径为：

`work_dirs/hgs_retina/ssdd_retinanet_r50_w6a6_ssi_lsq_200e_lr3e-3/epoch_90.pth`

该长跑的当前最佳验证结果是 epoch 90 的 mAP50=`0.616`，但由于训练未完成 200 epoch，不能称为 200 epoch 最终结果。

### 8.3 正确的消融顺序

后续每次只改变一个因素，并保持数据、模型、优化器、warmup、scheduler 和评估器完全一致：

1. FP32 baseline；
2. W6A6 + SSI + LSQ；
3. W4A4 + SSI + LSQ；
4. W4A4/W6A6 + FP32 bbox head（仅在必要时）；
5. SSI + LSQ + HGS compensation；
6. 如确有需要，再单独比较 SRQ。

每个实验必须记录 `mAP`、`mAP50`、`mAP75`、`AR`、最佳 epoch、学习率、bit-width、observer、fake quantizer、是否启用 HGS/SRQ 以及 workdir。

## 9. 其余历史 Workdir 的归档结论

以下历史目录已检查日志并归纳如下。它们不是当前 R50 统一 recipe 的正式结果，删除后仅保留本节摘要。

| Workdir | 模型/方法 | 日志中可见结果 | 归档判断 |
|---|---|---|---|
| `ssdd_retina_r50_fp32_100e` | R50 FP32 | epoch 100：mAP=`0.551`，mAP50=`0.890`，mAP75=`0.619` | 有效历史 FP32 参考，但训练 recipe 与当前 30 epoch 基线不同 |
| `ssdd_retina_r50_ssi_lsq_hgs_100e` | R50 SSI+LSQ+HGS | epoch 85：mAP=`0.235`，mAP50=`0.670`；日志后续无更高结果 | 历史 HGS 参考，不与当前基线直接比较 |
| `ssdd_retina_r18_fp32_100e_run` | R18 FP32 | epoch 100：mAP=`0.491`，mAP50=`0.857`，mAP75=`0.516` | 有效历史 R18 参考 |
| `ssdd_retina_r18_ssi_lsq_hgs_100e` | R18 SSI+LSQ+HGS | epoch 100：mAP=`0.250`，mAP50=`0.712`，mAP75=`0.078` | 历史 HGS 结果，精度明显低于对应 FP32 |
| `ssdd_retina_r18_ssi_lsq_hgs_fp32_head_100e` | R18 HGS，bbox head FP32 | epoch 95 最佳：mAP=`0.262`，mAP50=`0.728`；epoch 100：mAP=`0.263`，mAP50=`0.727` | head FP32 没有恢复到 R18 FP32 水平 |
| `ssdd_retina_r18_ssi_lsq_hgs_finetune_30e` | R18 HGS finetune | epoch 30：mAP=`0.296`，mAP50=`0.657`，mAP75=`0.215` | 仅可作为 finetune 参考 |
| `ssdd_retina_r18_ssi_lsq_srq_100e_run` | R18 SSI+LSQ+SRQ | epoch 85 最佳：mAP=`0.219`，mAP50=`0.681`；epoch 100：mAP=`0.213`，mAP50=`0.679` | SRQ 历史参考，不作为当前主线 |
| `ssdd_retina_r18_ssi_lsq_srq_hgs_200e` | R18 SSI+LSQ+SRQ+HGS | epoch 30 最佳：mAP50=`0.624`；epoch 35：mAP=`0.196`，mAP50=`0.643`；epoch 45：mAP=`0.178`，mAP50=`0.643` | 训练后期退化，不能作为当前 recipe 依据 |
| `ssdd_retina_r18_ssi_lsq_srq_hgs_1e` | R18 SRQ+HGS smoke | epoch 1：mAP=`0` | 无效 smoke run |
| `ssdd_retina_r18_fp32_100e` | R18 FP32 | 无完整验证结果可用 | 不保留权重，仅保留目录名称和用途 |
| `ssdd_retina_r18_ssi_lsq_srq_100e` | R18 SRQ | 无完整验证结果可用 | 不保留权重，仅保留目录名称和用途 |
| `qat_yolov8n_ssdd_200ep` | YOLOv8n QAT | epoch 54：mAP=`0.603`，mAP50=`0.895`，mAP75=`0.696` | 非 RetinaNet，仅作外部 QAT 参考 |
| `qat_faster_smoke` / `qat_faster_test` | Faster R-CNN QAT smoke/test | 无可用检测指标 | 仅调试产物 |

### 9.1 归档后的总体判断

- R50 历史 FP32 结果可以达到 mAP50=`0.890`，说明 R50 模型容量和数据流程没有根本问题。
- R50 SSI+LSQ+HGS 历史结果为 mAP50=`0.670`，但它与当前 recipe 的优化器、warmup、初始化和训练流程不完全一致，不能直接作为最终量化结论。
- R18 的历史结果显示，HGS、SRQ 和 FP32 head 的效果受训练 recipe 影响明显；不能把这些旧实验直接外推到当前 R50 W6A6/W4A4 实验。
- 当前正式结论仍以本文件第 2 节的 FP32 30 epoch 基线和正在运行的统一 recipe W6A6 为准。
