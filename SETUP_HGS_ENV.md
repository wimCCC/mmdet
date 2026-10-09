# 复现 HGS / SSDD 低比特 QAT 环境

这份文档 + `setup_env_hgs.sh` + `tools/dataset_converters/ssdd.py` 用来在一台新机器上
复现 `configs/_hgs_retina/` 下所有量化 config 的运行环境。

首次验证环境（2026-09-29）：

| 组件 | 版本 | 说明 |
|---|---|---|
| OS / GPU | Ubuntu 22.04 / RTX 4090 24 GB | gcc 11.4，nvcc 11.5 |
| Python | 3.10.21 | conda |
| torch | 2.0.1+cu117 | 必须 ≥2.0，但别超过 mqb-torch251 能承受的范围 |
| torchvision | 0.15.2+cu117 | 与 torch 严格配对 |
| mmcv | 2.0.1 | mmdet 3.0.0 要求 `>=2.0.0rc4,<2.1.0` |
| mmengine | 0.10.2 | |
| mmdet | 3.0.0 | 本仓库，editable |
| MQBench | 0.0.6 (`mqb-torch251` 分支) | **关键**，见坑 3 |
| numpy | 1.26.4 | **必须 <2**，见坑 2 |
| setuptools | 80.10.2 | **必须 <81**，见坑 1b |
| mqbench 数据源 | `hf-mirror.com/datasets/dronefreak/SSDD` | 见坑 5 |

## 快速开始

```bash
git clone <this-repo> && cd <this-repo>

# 1) 环境
bash setup_env_hgs.sh                    # 约 10 分钟

# 2) 数据集（约 50 MB，2 分钟）
conda activate hgs
python tools/dataset_converters/ssdd.py \
    --out data/SSDD/raw/Official-SSDD-OPEN/BBox_SSDD/coco_style

# 3) 训练
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    configs/_hgs_retina/ssdd_retinanet_r50_w6a6_ssi_lsq_200e_lr3e-3.py
```

`setup_env_hgs.sh` 会自己跑一遍 import 自检；下面的「验证清单」是更细的逐项检查。

> **换机器时 git 里有什么**：代码、全部 config、`setup_env_hgs.sh`、数据转换脚本，
> 以及两份文档（本文 + `HGS_W6A6_ABLATION_RESULTS.md`）。
> **不在 git 里**：`work_dirs/`（checkpoint 与训练日志，已在 `.gitignore`）、
> `data/`（用转换脚本重建，见坑 5）、`MQBench/`（未注册的 gitlink，要手动 clone，
> 见坑 3）。实验结果只以文档形式随 git 迁移，**work_dirs 需要的话得单独拷贝**。

## 六个坑

这几条都是实际踩到并定位过的，不是预防性提醒。

### 坑 1：mmcv 版本窗口很窄

mmdet 3.0.0 硬性要求 `mmcv>=2.0.0rc4,<2.1.0`。用 `mim install mmcv` 或装成 2.1+/2.2 会
直接报 registry / `Config` 相关错误。脚本用官方预编译轮子：

```bash
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
```

索引路径同时绑 CUDA 和 torch 版本（`cu117/torch2.0`），换 torch 版本要同步换索引。

### 坑 1b：setuptools ≥81 会让 torch 2.0.1 的 setup.py 装不上

`conda create python=3.10` 现在给的是 **setuptools 83**，而 setuptools 81 起**移除了
`pkg_resources`**。torch 2.0.1 的 `torch/utils/cpp_extension.py` 第 25 行仍然是：

```python
from pkg_resources import packaging  # type: ignore[attr-defined]
```

而本仓库 `setup.py` 顶部就 `from torch.utils.cpp_extension import ...`，于是
`pip install -e .` 直接挂在生成元数据阶段：

```
File "<string>", line 12, in <module>
  from pkg_resources import packaging
ModuleNotFoundError: No module named 'pkg_resources'
error: metadata-generation-failed
```

**加 `--no-build-isolation` 也救不了**——隔离构建只是换了个新版 setuptools 在隔离环境里
重复同样的失败。正确做法是降级：

```bash
pip install "setuptools<81"     # 必须排在 pip install -e . 之前
```

这条很隐蔽：如果你的环境碰巧是旧 setuptools（比如 80.x），一切正常；换台新机器从零建环境
就必挂。本次是干净环境复现测试才暴露出来的。

### 坑 2：numpy 会被 mmcv 顶到 2.x

mmcv 轮子的依赖里 numpy 没有上界，pip 会装成 2.x。torch 2.0.1 是针对 numpy 1.x 编译的，
import 时报：

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.0
```

**`pip install "numpy<2"` 必须在装完 mmcv 之后执行**，写在前面会被后续依赖解析覆盖掉。
（`opencv-python 5.x` 的元数据会声称需要 `numpy>=2`，那条冲突警告可以忽略，`import cv2`
实测正常。）

### 坑 3：MQBench 必须用 `mqb-torch251` 分支（最容易卡住的一条）

`MQBench/` 在本仓库里是个 gitlink（mode 160000），但**仓库里没有 `.gitmodules`**，所以
`git submodule update --init` 不会工作——目录会一直是空的。把它 clone 到该路径即可：

```bash
git clone --branch mqb-torch251 https://github.com/ModelTC/MQBench.git MQBench
pip install --no-deps --no-build-isolation -e ./MQBench
```

为什么不能用默认分支（`mqb-torch1.10`，也就是 0.0.6）：它的
`mqbench/fuser_method_mappings.py` 第 3 行

```python
from torch.quantization.fx.fusion_patterns import ConvBNReLUFusion, ModuleReLUFusion
```

这两个类在 torch 2.0 被删掉了（融合逻辑改到 `torch.ao.quantization.fx.fuse_handler.py`）。
报错发生在 `QATCompensationRunner.from_cfg` → `register_mqbench_fake_quantizers()`：

```
ImportError: cannot import name 'ConvBNReLUFusion' from 'torch.quantization.fx.fusion_patterns'
```

`mqb-torch251` 分支把这行注释掉并改用 `_get_custom_conv_configs`，而它仍依赖的
`torch.quantization.quantize_fx._swap_ff_with_fxff`、`torch.quantization.QConfig` 在
torch 2.0.1 里都还在，**所以不需要降级 torch**。

装的时候加 `--no-deps`：MQBench 的 `requirements.txt` 钉了自己的 torch/numpy，不加会把
上面装好的版本换掉。本仓库的 QAT 路径不需要它的 onnx/onnxruntime 那些依赖（只有
`tools/export_onnx.py` 需要）。

> GitHub 在部分网络下连不上（本机 clone 直接超时，`curl github.com` 却返回 200）。
> 可以 `MQBENCH_REPO=/path/to/local/MQBench bash setup_env_hgs.sh` 用本地克隆，
> 事后 `git remote set-url origin https://github.com/ModelTC/MQBench.git` 改回来。

### 坑 4：`vp_qod` 的 import

`mmdet/models/dense_heads/anchor_head.py:16` 有 `from vp_qod import HQOD_loss`，而
`vp_qod.py` 在**仓库根目录**、不在 mmdet 包内。跑 `python tools/train.py` 时 `sys.path[0]`
是 `tools/`，所以会 `ModuleNotFoundError: No module named 'vp_qod'`。

`SSDD_RETINANET_R50_QUANTIZATION_RECIPE.md` 的做法是每次导出
`PYTHONPATH=/data5/caiwm/mmdet`。脚本改成在 site-packages 里放一个 `.pth`，让环境自带：

```bash
echo "$REPO_ROOT" > "$(python -c 'import site; print(site.getsitepackages()[0])')/mmdet_repo_root.pth"
```

想恢复 `PYTHONPATH` 方式的话，删掉这个 `.pth` 即可。

### 坑 5：数据集要自己构建，且必须核对待划分

config 里原本写的是 `/data5/caiwm/mmdet/data/...`，那台机器的挂载点在别处不存在。
官方 SSDD（`TianwenZhang0825/Official-SSDD`）只提供 Google Drive / 百度网盘链接，
没法脚本化下载；Google Drive 在部分网络直接超时。

`tools/dataset_converters/ssdd.py` 改从 HuggingFace 镜像的 `dronefreak/SSDD` 取——
它有全部 1160 张官方图，且每个 split 带 `metadata.jsonl`，里面**已经是绝对坐标的 COCO
`xywh`**，所以不涉及 XML/YOLO 坐标转换。`hf-mirror.com` 可达而 `huggingface.co` 不可达，
脚本因此带 `--endpoint`。

**关键是划分要对上**：上游打包是 YOLO 的 789/139/232，而 config 要的是官方
**928 train / 232 test**，即 `train + valid = 928`。脚本跑完会核对：

```
train: 928 images, 2041 annotations (expected 928/2041) -> OK
test: 232 images, 546 annotations (expected 232/546) -> OK
```

`2041` 这个标注数与 `SSDD_RETINANET_R50_QUANTIZATION_RECIPE.md` 里记录的
「928 张图像、2041 个标注」一致，是划分正确的判据。数量不符时脚本会**主动报错退出**
（`--skip-count-check` 可跳过），因为错的划分会让新旧实验结果无法比较。

## 数据路径

`configs/_hgs_retina/_base_/datasets/ssdd.py` 现在按顺序取第一个存在的路径：

```python
_data_root_candidates = [
    '/data5/caiwm/mmdet/data/SSDD/raw/Official-SSDD-OPEN/BBox_SSDD/coco_style/',
    '/home2/caiwm/mmdet/data/SSDD/raw/Official-SSDD-OPEN/BBox_SSDD/coco_style/',
]
```

所以新机器上要么把数据放到 `/home2/caiwm/mmdet/data/...`（仓库内 `data/` 已被
`.gitignore` 忽略），要么改这个列表，要么直接用 `--cfg-options` 覆盖：

```bash
--cfg-options \
  train_dataloader.dataset.data_root=/your/path/ \
  val_dataloader.dataset.data_root=/your/path/ \
  test_dataloader.dataset.data_root=/your/path/ \
  val_evaluator.ann_file=/your/path/annotations/test.json \
  test_evaluator.ann_file=/your/path/annotations/test.json
```

## 验证清单

```bash
conda activate hgs
cd /tmp   # 离开仓库目录，避免 cwd 造成 shadow

# 1) 全部依赖 + CUDA 扩展
python -c "
import numpy, torch, torchvision, mmcv, mmengine, mmdet, mqbench, vp_qod
from mmcv.ops import nms, roi_align
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.fake_quantize import LearnableFakeQuantize, QDropFakeQuantize
from mqbench.observer import ClipStdObserver, ObserverBase
from mmdet.models.quantization import SSIClipObserver, HGSSolver
print(torch.__version__, torch.cuda.is_available(), mmcv.__version__, mqbench.__file__)
"

# 2) config 能解析、data_root 指到真实数据
python -c "
from mmengine.config import Config
import os.path as osp
cfg = Config.fromfile('/path/to/repo/configs/_hgs_retina/ssdd_retinanet_r50_w6a6_ssi_lsq_200e_lr3e-3.py')
print(cfg.train_dataloader.dataset.data_root, osp.isdir(cfg.train_dataloader.dataset.data_root))
"

# 3) 数据集能读、模型能建
cd /path/to/repo && python -c "
from mmengine.config import Config
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules
register_all_modules()
cfg = Config.fromfile('configs/_hgs_retina/ssdd_retinanet_r50_w6a6_ssi_lsq_200e_lr3e-3.py')
for k in ('train_dataloader','val_dataloader'):
    ds = DATASETS.build(cfg[k]['dataset'])
    print(k, len(ds), tuple(ds[0]['inputs'].shape))
"
# 期望：train_dataloader 928 (3, ~567, 800) / val_dataloader 232 (3, ~621, 800)
```

## 运行注意（都是实际踩过的）

- **训练进程必须脱离会话**：直接在终端里跑，SSH/终端会话一结束进程就被杀。
  实际发生过一次：`ft_60e_pt_s2`（seed 2 的 W6A6 per-tensor run）跑到 ep26 被杀，
  还没到 checkpoint 落盘周期（`val_interval`/`interval=10`），整个 run 作废。
  正确姿势：

  ```bash
  setsid nohup env CUDA_VISIBLE_DEVICES=0 python tools/train.py <config> \
      > /tmp/<run>.log 2>&1 &
  ```

  验证已脱离：`ps -o ppid= -p <pid>` 输出 `1`。
- **独占与共享 GPU 差 4 倍**：一个 60e run 独占 4090 约 40–48 分钟（116 iter/epoch，
  ~0.35 s/iter）；与他人共享时实测退化到 ~2.9 min/epoch（约 4 倍），排期要按最坏情况留。
- **评测里的 P/R 是本仓库新增的**：`val_evaluator` 同时挂 `CocoMetric` 与
  `PrecisionRecallMetric`（`mmdet/evaluation/metrics/precision_recall_metric.py`），
  日志表现为 `det/precision` / `det/recall` / `det/f1` / `det/score_thr`。
  **`coco/ship_precision` 不是 precision**——它恒等于 `coco/bbox_mAP`。
  阈值口径与读法见 `HGS_W6A6_ABLATION_RESULTS.md` §D2。

## 显存与其他

- `batch_size=8` + 800×800 输入约占 **12.9 GB**，24 GB 卡有充足余量；`batch_size=1` 约 2 GB。
- `work_dirs/` 与 `data/` 都已写进 `.gitignore`，训练产物不会进 git（每个 config 的
  `work_dir` 是相对路径，落在仓库内）。想放到别处用 `--work-dir`。
- `MQBench` 是未注册的 gitlink，`git status` 会一直显示它和记录的 commit 不一致，属正常。
- `SSDD_RETINANET_R50_QUANTIZATION_RECIPE.md` 提到避开 GPU 7。
- 训练脚本若报 `persistent_workers option needs num_workers > 0`，是 config 里
  `persistent_workers=True` 与 `num_workers=0` 冲突，把 `num_workers` 调回 4 即可。
