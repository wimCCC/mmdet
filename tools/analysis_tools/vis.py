# # Copyright (c) OpenMMLab
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# import os.path as osp
# import torch
# import cv2
# import numpy as np

# from mmengine.config import Config
# from mmengine.runner import load_checkpoint
# from mmengine.registry import MODELS
# from mmcv.transforms import Compose

# from mmdet.utils import register_all_modules
# register_all_modules()

# ####################################
# # 1. 路径配置
# ####################################
# config_file = '/home1/zhangyn/code/mmdetection-3.0.0/configs/_retinanet/ssdd_retina_r18.py'
# checkpoint_file = '/home1/zhangyn/code/mmdetection-3.0.0/tools/quanti/retinanet/SSDD/w8a8_new/dorefa_11_25_07_43/epoch_300.pth'

# img_dir = '/home1/zhangyn/code/mmdetection-3.0.0/data/SSDD/images/test/'
# save_dir = '/home1/zhangyn/code/mmdetection-3.0.0/tools/analysis_tools/vis_out'

# score_thr = 0.3
# os.makedirs(save_dir, exist_ok=True)

# ####################################
# # 2. 构建模型
# ####################################
# cfg = Config.fromfile(config_file)

# if 'backbone' in cfg.model:
#     cfg.model.backbone.init_cfg = None

# model = MODELS.build(cfg.model)
# load_checkpoint(model, checkpoint_file, map_location='cpu')

# model.cuda()
# model.eval()

# ####################################
# # 3. 安全构建 inference pipeline
# ####################################
# # 自动找 Resize 的 scale
# resize_scale = None
# for t in cfg.test_dataloader.dataset.pipeline:
#     if t['type'] == 'Resize':
#         resize_scale = t['scale']
#         break

# assert resize_scale is not None, '❌ Resize scale not found in pipeline'

# test_pipeline = Compose([
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', scale=resize_scale, keep_ratio=True),
#     dict(type='PackDetInputs')
# ])

# ####################################
# # 4. 推理 + 手动画框（终局）
# ####################################
# with torch.no_grad():
#     for name in os.listdir(img_dir):
#         if not name.lower().endswith(('.jpg', '.png', '.bmp', '.tif')):
#             continue

#         img_path = osp.join(img_dir, name)

#         data = dict(img_path=img_path)
#         # data = test_pipeline(data)

#         inputs = data['inputs'].unsqueeze(0).cuda()
#         data_samples = [data['data_samples']]

#         outputs = model(inputs)

#         # ===== 自动解包 RetinaNet 输出 =====
#         if isinstance(outputs, (list, tuple)):
#             cls_scores = outputs[0]
#             bbox_preds = outputs[1]
#         else:
#             raise TypeError(f'Unexpected model output type: {type(outputs)}')

#         results = model.bbox_head.predict_by_feat(
#             cls_scores,
#             bbox_preds,
#             img_metas=[data_samples[0].metainfo],
#             rescale=True
#         )[0]

#         bboxes = results.bboxes.cpu().numpy()
#         scores = results.scores.cpu().numpy()

#         keep = scores > score_thr
#         bboxes = bboxes[keep]

#         img = cv2.imread(img_path)
#         if img is None:
#             continue

#         for box in bboxes:
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         cv2.imwrite(osp.join(save_dir, name), img)

# print(f'✅ 可视化完成，结果保存在：{save_dir}')



# ============================================
# Quantized RetinaNet Visualization Script
# For MMDetection 3.x
# ============================================

import os
import sys
import cv2
import torch
from tqdm import tqdm

# ===== 基础环境 =====
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append('/home1/zhangyn/code/mmdetection-3.0.0')

# ===== 关键：初始化 mmdet registry =====
import mmdet.models  # 必须

# ===== 如果你有自定义量化 detector，一定要 import =====
from mmdet.models.detectors import retinanet_quanti  # 路径按你实际的来

from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint
from mmdet.structures import DetDataSample
from mmdet.datasets.transforms import PackDetInputs
from mmdet.utils import register_all_modules

register_all_modules()

# ================== 路径配置 ==================
CONFIG_FILE = '/home1/zhangyn/code/mmdetection-3.0.0/configs/_retinanet/ssdd_retina_r18.py'
CHECKPOINT = '/home1/zhangyn/code/mmdetection-3.0.0/tools/quanti/retinanet/SSDD/w8a8_new/dorefa_11_25_07_43/epoch_300.pth'

IMG_DIR = '/home1/zhangyn/code/mmdetection-3.0.0/data/SSDD/images/test'
OUT_DIR = '/home1/zhangyn/code/mmdetection-3.0.0/tools/analysis_tools/vis_out'

SCORE_THR = 0.3

os.makedirs(OUT_DIR, exist_ok=True)

# ================== 构建模型 ==================
cfg = Config.fromfile(CONFIG_FILE)

# 兼容 mmdet 3.x
if 'pretrained' in cfg.model:
    cfg.model.pretrained = None

model = MODELS.build(cfg.model)
model.eval().cuda()

load_checkpoint(model, CHECKPOINT, map_location='cpu')

# ================== 预处理 ==================
mean = torch.tensor(cfg.data_preprocessor.mean).view(1, -1, 1, 1).cuda()
std = torch.tensor(cfg.data_preprocessor.std).view(1, -1, 1, 1).cuda()

def preprocess(img):
    img = img.astype('float32')
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
    img = (img - mean) / std
    return img

# ================== 推理 + 可视化 ==================
with torch.no_grad():
    for name in tqdm(os.listdir(IMG_DIR)):
        if not name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            continue

        img_path = os.path.join(IMG_DIR, name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        inp = preprocess(img)

        data_sample = DetDataSample()
        data_sample.set_metainfo(
            dict(
                img_shape=(h, w),
                ori_shape=(h, w),
                scale_factor=1.0
            )
        )

        results = model.test_step(
            dict(inputs=inp, data_samples=[data_sample])
        )[0]

        if not hasattr(results, 'pred_instances'):
            cv2.imwrite(os.path.join(OUT_DIR, name), img)
            continue

        bboxes = results.pred_instances.bboxes.cpu().numpy()
        scores = results.pred_instances.scores.cpu().numpy()
        labels = results.pred_instances.labels.cpu().numpy()

        for box, score, label in zip(bboxes, scores, labels):
            if score < SCORE_THR:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f'{label}:{score:.2f}',
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        cv2.imwrite(os.path.join(OUT_DIR, name), img)

print(f'✅ 可视化完成，结果保存在：{OUT_DIR}')



# ############################
# # 1. 路径配置（只改这里）
# ############################
# config_file = '/home1/zhangyn/code/mmdetection-3.0.0/configs/_retinanet/ssdd_retina_r18.py'
# checkpoint_file = '/home1/zhangyn/code/mmdetection-3.0.0/tools/quanti/retinanet/SSDD/w8a8_new/dorefa_11_25_07_43/epoch_300.pth'

# img_dir = '/home1/zhangyn/code/mmdetection-3.0.0/data/SSDD/images/test/'
# save_dir = '/home1/zhangyn/code/mmdetection-3.0.0/tools/analysis_tools/vis_out'

# score_thr = 0.3

# os.makedirs(save_dir, exist_ok=True)

# ############################
# # 2. 构建模型（mmdet 3.x 正确方式）
# ############################
# cfg = Config.fromfile(config_file)

# # ❗ 关键：mmdet 3.x 不能有 pretrained
# if 'backbone' in cfg.model:
#     cfg.model.backbone.init_cfg = None

# cfg.model.test_cfg.rcnn = None if 'rcnn' in cfg.model else None

# model = MODELS.build(cfg.model)
# load_checkpoint(model, checkpoint_file, map_location='cpu')

# model.eval()
# model.cuda()

# ############################
# # 3. 构建测试 pipeline
# ############################
# test_pipeline = cfg.test_dataloader.dataset.pipeline
# test_pipeline = Compose(test_pipeline)

# ############################
# # 4. 可视化器
# ############################
# visualizer = VISUALIZERS.build(
#     dict(
#         type='DetLocalVisualizer',
#         vis_backends=[dict(type='LocalVisBackend')],
#         name='visualizer'
#     )
# )
# visualizer.dataset_meta = dict(classes=('ship',))

# ############################
# # 5. 推理 + 画框
# ############################
# with torch.no_grad():
#     for img_name in os.listdir(img_dir):
#         if not img_name.lower().endswith(('.jpg', '.png', '.bmp', '.tif')):
#             continue

#         img_path = osp.join(img_dir, img_name)
#         data = dict(img_path=img_path)
#         data = test_pipeline(data)

#         data['inputs'] = data['inputs'].unsqueeze(0).cuda()
#         data['data_samples'] = [data['data_samples']]

#         outputs = model.test_step(data)[0]

#         # 过滤低分框
#         keep = outputs.pred_instances.scores > score_thr
#         outputs.pred_instances = outputs.pred_instances[keep]

#         visualizer.add_datasample(
#             name=img_name,
#             image=mmcv.imread(img_path, channel_order='rgb'),
#             data_sample=outputs,
#             draw_gt=False,
#             show=False,
#             out_file=osp.join(save_dir, img_name)
#         )

# print(f'✔ 可视化完成，结果保存在：{save_dir}')
