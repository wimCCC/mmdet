#!/bin/bash
# Reproducible conda environment for the HGS / SSDD low-bit QAT configs.
#
# Verified on 2026-09-29 with:
#   Ubuntu 22.04, gcc 11.4, nvcc 11.5, RTX 4090, python 3.10.21
#   torch 2.0.1+cu117 / torchvision 0.15.2+cu117 / mmcv 2.0.1 / mmengine 0.10.2
#   mmdet 3.0.0 (this repo, editable) / MQBench 0.0.6 (mqb-torch251 branch)
#
# Usage:
#   bash setup_env_hgs.sh                # env name "hgs", repo = script location
#   ENV_NAME=myenv bash setup_env_hgs.sh
#
# The four things that silently break this setup, in order of how much time
# they cost to diagnose, are documented inline as [PITFALL n].
set -euo pipefail

ENV_NAME="${ENV_NAME:-hgs}"
PY_VERSION="${PY_VERSION:-3.10}"
# Must stay consistent: the mmcv wheel index is keyed on both CUDA and torch.
TORCH_VERSION="${TORCH_VERSION:-2.0.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.15.2}"
CUDA_TAG="${CUDA_TAG:-cu117}"          # used for the pytorch wheel index
MMCV_VERSION="${MMCV_VERSION:-2.0.1}"
MMENGINE_VERSION="${MMENGINE_VERSION:-0.10.2}"
MMCV_INDEX="${MMCV_INDEX:-https://download.openmmlab.com/mmcv/dist/${CUDA_TAG}/torch2.0/index.html}"
# GitHub is unreachable from some networks; point this at a local clone instead.
MQBENCH_REPO="${MQBENCH_REPO:-https://github.com/ModelTC/MQBench.git}"
MQBENCH_BRANCH="${MQBENCH_BRANCH:-mqb-torch251}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "==> repo root : $REPO_ROOT"
echo "==> env name  : $ENV_NAME (python $PY_VERSION)"

# --- conda env -------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    for cand in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
        if [ -f "$cand/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1091
            source "$cand/etc/profile.d/conda.sh"
            break
        fi
    done
fi
command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda not found. Install miniconda first." >&2; exit 1; }

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> conda env '$ENV_NAME' already exists; reusing it"
else
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi
conda activate "$ENV_NAME"

# --- pytorch ---------------------------------------------------------------
# torch 2.0.1 is the newest release the mqb-torch251 MQBench branch is known to
# work with here, and it still ships the quantization FX entry points that
# branch imports (`torch.quantization.quantize_fx._swap_ff_with_fxff`,
# `torch.quantization.QConfig`).
echo "==> installing torch ${TORCH_VERSION}+${CUDA_TAG} / torchvision ${TORCHVISION_VERSION}"
pip install \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# --- mmcv / mmengine -------------------------------------------------------
# [PITFALL 1] mmdet 3.0.0 pins mmcv to >=2.0.0rc4,<2.1.0. Installing a newer
# mmcv (or letting mim pick one) breaks the registry API.
echo "==> installing mmcv ${MMCV_VERSION} from prebuilt wheel index"
pip install "mmcv==${MMCV_VERSION}" -f "$MMCV_INDEX"
pip install "mmengine==${MMENGINE_VERSION}"

# [PITFALL 2] The mmcv wheel declares numpy without an upper bound, so pip
# upgrades to numpy 2.x. torch 2.0.1 was compiled against numpy 1.x and fails
# at import with "A module that was compiled using NumPy 1.x cannot be run in
# NumPy 2.0.0". This pin MUST come after the mmcv install, not before.
echo "==> pinning numpy<2 (must happen after mmcv install)"
pip install "numpy<2"

# --- mmdet (this repo) -----------------------------------------------------
# [PITFALL 1b] conda's python 3.10 ships setuptools >= 81, which dropped the
# `pkg_resources` module. torch 2.0.1's torch/utils/cpp_extension.py still does
# `from pkg_resources import packaging` at import time, and setup.py imports
# that module, so `pip install -e .` fails with:
#     ModuleNotFoundError: No module named 'pkg_resources'
# Under build isolation you get the same failure, just with a newer setuptools
# inside the isolated env. Pin below 81 instead.
echo "==> pinning setuptools<81 (torch 2.0.1 setup.py needs pkg_resources)"
pip install "setuptools<81"

echo "==> installing mmdet from $REPO_ROOT (editable)"
pip install -v -e . --no-build-isolation

# --- MQBench ---------------------------------------------------------------
# [PITFALL 3] The default / mqb-torch1.10 MQBench imports
#     from torch.quantization.fx.fusion_patterns import ConvBNReLUFusion
# which torch >= 2.0 removed, so `register_mqbench_fake_quantizers()` raises
# ImportError at runner build time. Only the mqb-torch251 branch survives.
if [ -d "$REPO_ROOT/MQBench/mqbench" ]; then
    echo "==> reusing existing MQBench checkout at $REPO_ROOT/MQBench"
    git -C "$REPO_ROOT/MQBench" rev-parse --abbrev-ref HEAD | grep -qx "$MQBENCH_BRANCH" \
        || echo "    WARNING: not on $MQBENCH_BRANCH; check 'git -C MQBench log -1'"
else
    echo "==> cloning MQBench ($MQBENCH_BRANCH) into $REPO_ROOT/MQBench"
    git clone --branch "$MQBENCH_BRANCH" "$MQBENCH_REPO" "$REPO_ROOT/MQBench" \
        || {
            echo "ERROR: MQBench clone failed. GitHub may be blocked here." >&2
            echo "       Clone it elsewhere and re-run with:" >&2
            echo "       MQBENCH_REPO=/path/to/local/MQBench bash $0" >&2
            exit 1
        }
fi
# --no-deps on purpose: MQBench's requirements.txt pins its own torch/numpy,
# and pip would otherwise replace the versions installed above. The QAT path
# used by these configs does not need its onnx/onnxruntime extras.
pip install --no-deps --no-build-isolation -e "$REPO_ROOT/MQBench"

# --- vp_qod import shim ----------------------------------------------------
# [PITFALL 4] mmdet/models/dense_heads/anchor_head.py does `from vp_qod import
# HQOD_loss`, and vp_qod.py sits at the repo root rather than inside the mmdet
# package. Running `python tools/train.py ...` puts tools/ (not the repo root)
# on sys.path, so the import fails with ModuleNotFoundError unless PYTHONPATH
# is exported. A .pth in site-packages makes the env self-sufficient instead.
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo "$REPO_ROOT" > "$SITE_PACKAGES/mmdet_repo_root.pth"
echo "==> wrote $SITE_PACKAGES/mmdet_repo_root.pth"

# --- verify ----------------------------------------------------------------
echo
echo "==> verifying"
cd /tmp   # leave the repo dir so nothing is shadowed by the cwd
python - <<'PY'
import numpy, torch, torchvision, mmcv, mmengine, mmdet, mqbench, vp_qod
from mmcv.ops import nms, roi_align          # proves the CUDA extensions built
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.fake_quantize import LearnableFakeQuantize, QDropFakeQuantize
from mqbench.observer import ClipStdObserver, ObserverBase
from mmdet.models.quantization import SSIClipObserver, HGSSolver

print('numpy      ', numpy.__version__)
print('torch      ', torch.__version__, '| cuda', torch.cuda.is_available())
print('torchvision', torchvision.__version__)
print('mmcv       ', mmcv.__version__)
print('mmengine   ', mmengine.__version__)
print('mmdet      ', mmdet.__version__)
print('mqbench    ', mqbench.__file__)
print('vp_qod     ', vp_qod.__file__)
print('ALL IMPORTS OK')
PY

echo
echo "==> done. Next: build the dataset, then train."
echo "    python tools/dataset_converters/ssdd.py --out data/SSDD/raw/Official-SSDD-OPEN/BBox_SSDD/coco_style"
echo "    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/_hgs_retina/ssdd_retinanet_r50_w6a6_ssi_lsq_200e_lr3e-3.py"
