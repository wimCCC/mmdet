# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

from torch import Tensor
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector, BaseDetector_quanti, BaseDetector_quanti_retinanet, BaseDetector_quanti_fcos, BaseDetector_quanti_fcos_2, BaseDetector_yolo_kd
import torch
import torch.nn.functional as F


def unpack_gt_instances(batch_data_samples: SampleList) -> tuple:
    batch_gt_instances = []
    batch_gt_instances_ignore = []
    batch_img_metas = []
    for data_sample in batch_data_samples:
        batch_img_metas.append(data_sample.metainfo)
        batch_gt_instances.append(data_sample.gt_instances)
        if 'ignored_instances' in data_sample:
            batch_gt_instances_ignore.append(data_sample.ignored_instances)
        else:
            batch_gt_instances_ignore.append(None)

    return batch_gt_instances, batch_gt_instances_ignore, batch_img_metas

@MODELS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x


@MODELS.register_module()
class SingleStageDetector_yolo_kd(BaseDetector_yolo_kd):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.adaptation_layers = nn.ModuleList([
                nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),
        ])
        
        self.channel_wise_adaptation = nn.ModuleList([
                nn.Linear(128, 128),
                nn.Linear(256, 256),
                nn.Linear(512, 512),
        ])
        
        self.spatial_wise_adaptation = nn.ModuleList([
                nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
        ])
        
        self.backbone_adaptation = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def get_masks(self, feat_maps, gt_bboxes, img, area_threshold_small_object):
        gt_bboxes = [gt_bbox.cuda() for gt_bbox in gt_bboxes]
        # 获取每个特征图的尺寸（高和宽）
        featmap_sizes = [featmap.size()[-2:] for featmap in feat_maps] # [torch.Size([76, 128]), torch.Size([38, 64]), torch.Size([19, 32]), torch.Size([10, 16]), torch.Size([5, 8])]
        # 获取设备信息，通常是GPU
        device = feat_maps[0].device
        # 获取原始图像的尺寸
        img_ori = img.size()[-2:]
        img_ori = torch.tensor([img_ori])   # 接近 [600, 1000]
        # 获取锚点（anchors）和有效标志（valid flags）列表
        # anchor_list, valid_flag_list = self.bbox_head.get_anchors(featmap_sizes, img_metas, device=device)
        # 初始化一个列表，用于存储每个批次图像的掩码
        mask_batch = []

        # 获取批次大小
        batch_size = len(gt_bboxes)
        # 获取特征图的层级数
        levels = len(feat_maps)
        # 遍历每个图像
        area_sum = 0.0
        area_min = 10000.0
        area_max = -1.0
        num_target = 0
        for i in range(batch_size):
            # 初始化一个列表，用于存储每个图像的掩码
            mask_per_im = []
            # 计算真实边界框的面积
            area = torch.sqrt((gt_bboxes[i][:, 2] - gt_bboxes[i][:, 0]) * (gt_bboxes[i][:, 3] - gt_bboxes[i][:, 1]))
            area_avg = torch.sum(area) / area.size(0)
            area = area.cuda()
            area_avg = area_avg.cuda()
                
            # 计算原始图像的面积
            area_image = torch.sqrt((img_ori[:, 0] * img_ori[:, 1]).to(torch.float32)).cuda()
            # 确定每个边界框应该分配给哪个特征图层级
            target_lvls = torch.floor(4 + torch.log2(area / area_image + 1e-6)).cuda()
            # 将层级限制在0到4之间
            target_lvls = target_lvls.clamp(min=0, max=4).long()
            # 计算面积权重
            area_weight = torch.exp(-area / (area_image / 2)) + 1   # 默认，但值都比较大，基本都是1.5以上，区分度不高
            area_weight_2 = area_avg / area
            # 当前图像都是小目标，那就对小于阈值的目标权重+0.5予以特别关注
            if area_avg < area_threshold_small_object:
                flag = area < area_threshold_small_object
                area_weight_2[flag] += 0.5
            # 遍历每个特征图层级
            for j in range(levels):
                # 获取当前层级的高和宽
                height, width = featmap_sizes[j][0], featmap_sizes[j][1]
                # 初始化当前层级的掩码，初始值为0
                mask_per_level = torch.zeros([height, width], dtype=torch.float32).cuda()
                # 获取当前特征图的尺寸
                fea_scale = featmap_sizes[j]
                # 将特征图尺寸转换为tensor
                fea_scale = torch.tensor([fea_scale])
                # 计算特征图相对于原始图像的缩放比例
                percent = torch.div(fea_scale.float(), img_ori.float()).cuda()
                # 初始化一个用于存放缩放后的边界框的tensor
                gt_trans = torch.zeros(gt_bboxes[i].shape, dtype=torch.float32).cuda()
                # 将边界框按比例缩放到对应的特征图尺寸
                gt_trans[:, :2] = gt_bboxes[i][:, :2] * percent
                gt_trans[:, 2:] = gt_bboxes[i][:, 2:] * percent
                # 初始化一个与边界框尺寸相同的掩码
                gt_mask = torch.zeros(gt_bboxes[i].shape, dtype=torch.float32).cuda()
                # 为当前层级的每个边界框生成掩码
                for k in range(gt_bboxes[i].shape[0]):
                    # 如果边界框分配给当前层级，则将掩码设置为1
                    if target_lvls[k] == j:
                        gt_mask[k, :] = 1
                # 将掩码和缩放后的边界框相乘，得到当前层级的边界框
                gt_trans = gt_trans * gt_mask
                # 遍历每个边界框
                for k in range(gt_bboxes[i].shape[0]):
                    # 如果边界框为空，则跳过
                    if torch.sum(gt_bboxes[i][k]) == 0.:
                        break
                    # 为每个边界框生成特征图尺寸的掩码
                    mask_per_gt = torch.zeros(height, width, dtype=torch.float32).cuda()
                    # 将掩码中边界框所在位置的值设置为对应的加权权重
                    mask_per_gt[gt_trans[k][1].int():gt_trans[k][3].int(), gt_trans[k][0].int():gt_trans[k][2].int()] = \
                        area_weight_2[k]
                    # 将当前边界框的掩码叠加到当前层级的掩码上
                    mask_per_level += mask_per_gt
                epsilon = 1e-6  # 一个小的常数，用于平滑掩码防止梯度消失
                mask_per_level = mask_per_level.float()
                mask_per_level = epsilon + (1 - epsilon) * mask_per_level
                # 将当前层级的掩码添加到图像的掩码列表中
                mask_per_im.append(mask_per_level)
            # 将图像的掩码列表添加到批次的掩码列表中
            mask_batch.append(mask_per_im)

        return mask_batch
  
    def calculate_kd_loss(self, feat_t, feat_s, _i, b, kd_feat_losses, kd_channel_losses, kd_spatial_losses, cfg):
        # 通道注意力参数
        c_t = 0.1
        c_s_ratio = 1.0
        # 空间注意力参数
        t = 0.1
        s_ratio = 1.0
        # 计算教师特征图的空间注意力图
        t_attention_mask = torch.mean(torch.abs(feat_t), dim=1, keepdim=True)
        size = t_attention_mask.size()
        t_attention_mask = torch.softmax(t_attention_mask.view(1, -1) / t, dim=1)
        t_attention_mask = t_attention_mask.view(size)

        # 计算学生特征图的空间注意力图
        s_attention_mask = torch.mean(torch.abs(feat_s), dim=1, keepdim=True)
        size = s_attention_mask.size()
        s_attention_mask = torch.softmax(s_attention_mask.view(1, -1) / t, dim=1)
        s_attention_mask = s_attention_mask.view(size)

        # 计算教师特征图的通道注意力图
        c_t_attention_mask = torch.mean(torch.abs(feat_t), dim=[2, 3], keepdim=True)
        size = c_t_attention_mask.size()
        c_t_attention_mask = torch.softmax(c_t_attention_mask.view(1, -1) / c_t, dim=1)
        c_t_attention_mask = c_t_attention_mask.view(size)

        # 计算学生特征图的通道注意力图
        c_s_attention_mask = torch.mean(torch.abs(feat_s), dim=[2, 3], keepdim=True)
        size = c_s_attention_mask.size()
        c_s_attention_mask = torch.softmax(c_s_attention_mask.view(1, -1) / c_t, dim=1)
        c_s_attention_mask = c_s_attention_mask.view(size)

        # 计算加权特征图损失
        sum_attention_mask = (t_attention_mask + s_attention_mask * s_ratio) / (1 + s_ratio)
        c_sum_attention_mask = (c_t_attention_mask + c_s_attention_mask * c_s_ratio) / (1 + c_s_ratio)
        
        diff = self.distL2(feat_t, feat_s, attention_mask=sum_attention_mask,
                                channel_attention_mask=c_sum_attention_mask, weight=1.0)
        # kd_feat_loss_cur = (torch.sum(diff) ** 0.5) * 1e-1 * 0.6
        # 前景背景解耦后使用的权重
        kd_feat_loss_cur = diff * 1e-1 * cfg.weight_feat_loss   # 0.5
        kd_feat_losses[b] += kd_feat_loss_cur
        
        # 计算通道损失
        kd_channel_loss_cur = torch.dist(torch.mean(feat_t, [2, 3]),
                                                    self.channel_wise_adaptation[_i](torch.mean(feat_s, [2, 3]))) * 4e-3 * cfg.weight_channel_loss  # 2.8
        kd_channel_losses[b] += kd_channel_loss_cur

        # # 计算空间损失
        t_spatial_pool = torch.mean(feat_t, [1]).view(1, 1, feat_t.size(2), feat_t.size(3))
        s_spatial_pool = torch.mean(feat_s, [1]).view(1, 1, feat_s.size(2), feat_s.size(3))
        kd_spatial_loss_cur = torch.dist(t_spatial_pool, self.spatial_wise_adaptation[_i](s_spatial_pool)) * 4e-3 * cfg.weight_spatial_loss # 3.3 
        
        kd_spatial_losses[b] += kd_spatial_loss_cur
    
    def calculate_kd_loss_weight(self, feat_t, feat_s, _i, b, kd_feat_losses, kd_channel_losses, kd_spatial_losses, cfg, mask_weight=None):
        # 通道注意力参数
        c_t = 0.1
        c_s_ratio = 1.0
        # 空间注意力参数
        t = 0.1
        s_ratio = 1.0
        # 计算教师特征图的空间注意力图
        t_attention_mask = torch.mean(torch.abs(feat_t), dim=1, keepdim=True)
        size = t_attention_mask.size()
        t_attention_mask = torch.softmax(t_attention_mask.view(1, -1) / t, dim=1)
        t_attention_mask = t_attention_mask.view(size)

        # 计算学生特征图的空间注意力图
        s_attention_mask = torch.mean(torch.abs(feat_s), dim=1, keepdim=True)
        size = s_attention_mask.size()
        s_attention_mask = torch.softmax(s_attention_mask.view(1, -1) / t, dim=1)
        s_attention_mask = s_attention_mask.view(size)

        # 计算教师特征图的通道注意力图
        c_t_attention_mask = torch.mean(torch.abs(feat_t), dim=[2, 3], keepdim=True)
        size = c_t_attention_mask.size()
        c_t_attention_mask = torch.softmax(c_t_attention_mask.view(1, -1) / c_t, dim=1)
        c_t_attention_mask = c_t_attention_mask.view(size)

        # 计算学生特征图的通道注意力图
        c_s_attention_mask = torch.mean(torch.abs(feat_s), dim=[2, 3], keepdim=True)
        size = c_s_attention_mask.size()
        c_s_attention_mask = torch.softmax(c_s_attention_mask.view(1, -1) / c_t, dim=1)
        c_s_attention_mask = c_s_attention_mask.view(size)

        # 计算加权特征图损失
        sum_attention_mask = (t_attention_mask + s_attention_mask * s_ratio) / (1 + s_ratio)
        c_sum_attention_mask = (c_t_attention_mask + c_s_attention_mask * c_s_ratio) / (1 + c_s_ratio)
        
        diff = self.distL2_mask(feat_t, feat_s, attention_mask=sum_attention_mask,
                                channel_attention_mask=c_sum_attention_mask, weight=1.0, mask_weight=mask_weight)
        # kd_feat_loss_cur = (torch.sum(diff) ** 0.5) * 1e-1 * 0.6
        # 前景背景解耦后使用的权重
        kd_feat_loss_cur = diff * 1e-1 * cfg.weight_feat_loss   # 0.5
        kd_feat_losses[b] += kd_feat_loss_cur
        
        # 计算通道损失
        kd_channel_loss_cur = torch.dist(torch.mean(feat_t, [2, 3]),
                                                    self.channel_wise_adaptation[_i](torch.mean(feat_s, [2, 3]))) * 4e-3 * cfg.weight_channel_loss  # 2.8
        kd_channel_losses[b] += kd_channel_loss_cur

        # # 计算空间损失
        t_spatial_pool = torch.mean(feat_t, [1]).view(1, 1, feat_t.size(2), feat_t.size(3))
        s_spatial_pool = torch.mean(feat_s, [1]).view(1, 1, feat_s.size(2), feat_s.size(3))
        kd_spatial_loss_cur = torch.dist(t_spatial_pool, self.spatial_wise_adaptation[_i](s_spatial_pool)) * 4e-3 * cfg.weight_spatial_loss # 3.3 
        
        kd_spatial_losses[b] += kd_spatial_loss_cur
    
    def distL2(self, tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None, weight=1.0):
        diff = (tensor_a - tensor_b) ** 2
        diff = diff * attention_mask
        diff = diff * channel_attention_mask
        diff = torch.sum(diff) ** 0.5
        return diff

    def distL2_mask(self, tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None, weight=1.0, mask_weight = None):
        diff = (tensor_a - tensor_b) ** 2
        diff = diff * attention_mask
        diff = diff * channel_attention_mask
        diff = diff * mask_weight
        diff = torch.sum(diff) ** 0.5
        return diff
   
    def cosine_distance(self, pred, eps=1e-8):
        pred_norm = F.normalize(pred, p=2, dim=1)  # Normalize each vector by its L2 norm.
        cos_sim = torch.mm(pred_norm, pred_norm.t())  # Compute the cosine similarity matrix.
        cos_sim = cos_sim.clamp(min=-1 + eps, max=1 - eps)  # Ensure values are in [-1, 1].
        cos_dist = 1 - cos_sim  # Convert to cosine distance.
        cos_dist.fill_diagonal_(0)  # Set diagonal to zero.
        return cos_dist
   
    def weighted_cosine_similarity(self, x1, x2, weights):
        # 加权特征向量
        weighted_x1 = x1 * weights
        weighted_x2 = x2 * weights
        
        # L2 归一化加权特征向量
        weighted_x1_norm = F.normalize(weighted_x1, p=2, dim=1)
        weighted_x2_norm = F.normalize(weighted_x2, p=2, dim=1)
        
        # 计算加权余弦相似度
        similarity = (weighted_x1_norm * weighted_x2_norm).sum(dim=1)
        
        return similarity
   
    def distance_wise_rkd_loss_new(self, foreground_s, background_s, foreground_t, background_t, kd_realtion_loss=None, cfg=None):
        # 展平特征图以便于计算余弦相似度
        weights_s = torch.where(foreground_s >= 1e-6, torch.tensor(1e7).cuda(), torch.tensor(1.0).cuda())
        weights_t = torch.where(foreground_t >= 1e-6, torch.tensor(1e7).cuda(), torch.tensor(1.0).cuda())
        weights_s_flat = weights_s.view(weights_s.size(0), weights_s.size(1), -1)
        weights_t_flat = weights_t.view(weights_t.size(0), weights_t.size(1), -1)
        fg_s_flat = foreground_s.view(foreground_s.size(0), foreground_s.size(1), -1) # [batch_size, channels, height*width]
        bg_s_flat = background_s.view(background_s.size(0), background_s.size(1), -1)
        fg_t_flat = foreground_t.view(foreground_t.size(0), foreground_t.size(1), -1)
        bg_t_flat = background_t.view(background_t.size(0), background_t.size(1), -1)
            
        # 计算前景和背景之间的余弦相似度
        fg_bg_sim_s = self.weighted_cosine_similarity(fg_s_flat.permute(0, 2, 1), bg_s_flat.permute(0, 2 ,1), weights_s_flat.permute(0, 2 ,1)) # [batch_size * height*width]
        fg_bg_sim_t = self.weighted_cosine_similarity(fg_t_flat.permute(0 ,2 ,1), bg_t_flat.permute(0 ,2 ,1), weights_t_flat.permute(0, 2 ,1))
        
        # 计算师生网络背景与背景之间的关系
        bg_relation_s = self.cosine_distance(background_s.view(background_s.shape[0], -1))
        bg_relation_t = self.cosine_distance(background_t.view(background_t.shape[0], -1))

        # 计算两种关系损失
        fg_bg_loss = F.smooth_l1_loss(fg_bg_sim_s, fg_bg_sim_t) 
        bg_bg_loss = F.smooth_l1_loss(bg_relation_s, bg_relation_t)

        # 将这两种损失加到总损失中
        kd_realtion_loss += (fg_bg_loss + bg_bg_loss) * cfg.weight_relation_loss
   
    def distance_wise_rkd_loss(self, feat_s, feat_t, kd_realtion_loss=None, cfg=None):
        feat_s = feat_s.view(feat_s.shape[0], -1)
        feat_t = feat_t.view(feat_t.shape[0], -1)

        # Normalize the predictions if required
        feat_s = F.normalize(feat_s, p=2, dim=1)
        feat_t = F.normalize(feat_t, p=2, dim=1)

        # Calculate cosine distances
        d_T = self.cosine_distance(feat_t)
        d_S = self.cosine_distance(feat_s)

        # Calculate the RKD distance loss
        loss = F.smooth_l1_loss(d_S, d_T) * cfg.weight_relation_loss
        kd_realtion_loss += loss
    
    def caculate_loss(self, feat_t, feat_s, losses=None, cfg=None, gt_bboxes=None, img=None):
        
        mask_batch = self.get_masks(feat_s, gt_bboxes, img, 60)
        
        transposed_mask_batch = [list(group) for group in zip(*mask_batch)]   
        for tensor_list in transposed_mask_batch:
            for i, tensor in enumerate(tensor_list):
                greater_than_threshold = tensor > 1e-6
                tensor_list[i] = tensor.masked_fill(greater_than_threshold, 1.0)
        
        inverted_transposed_mask_batch = []
        for tensor_list in transposed_mask_batch:
            inverted_tensor_list = []
            for tensor in tensor_list:
                inverted_tensor = torch.where(tensor == 1e-6, torch.ones_like(tensor), torch.full_like(tensor, 1e-6))
                inverted_tensor_list.append(inverted_tensor)
            inverted_transposed_mask_batch.append(inverted_tensor_list)
        
        batch_size = feat_s[0].size(0)
        
        kd_feat_losses = [0] * batch_size
        kd_channel_losses = [0] * batch_size
        kd_spatial_losses = [0] * batch_size

        feat_loss_record = torch.tensor(0.).cuda()
        realtion_loss_record = torch.tensor(0.).cuda()
        channel_loss_record = torch.tensor(0.).cuda()
        spatial_loss_record = torch.tensor(0.).cuda()
        
        # 先计算关系loss
        for i in range(len(feat_t)):
            x_aligned = self.adaptation_layers[i](feat_s[i])        
            foreground_s = torch.stack([x_aligned[m] * transposed_mask_batch[i][m] for m in range(x_aligned.size(0))])
            background_s = torch.stack([x_aligned[m] * inverted_transposed_mask_batch[i][m] for m in range(x_aligned.size(0))])
            foreground_t = []
            for m in range(feat_t[i].size(0)):
                foreground_t.append(feat_t[i][m] * transposed_mask_batch[i][m])
            foreground_t = torch.stack(foreground_t)
            background_t = []
            for m in range(feat_t[i].size(0)):
                background_t.append(feat_t[i][m] * inverted_transposed_mask_batch[i][m])
            background_t = torch.stack(background_t)
            self.distance_wise_rkd_loss_new(foreground_s, background_s, foreground_t, background_t, realtion_loss_record, cfg)

        
        # feat_t_foreground = feat_t.detach().clone()
        # feat_t_foreground = copy.deepcopy(feat_t_foreground)
        # feat_t_foreground = copy.deepcopy(feat_t)
        # feat_t_background = copy.deepcopy(feat_t)
        # 遍历每个样本
        for b in range(batch_size):
            # 对于每个样本，遍历所有尺度的特征图
            for _i in range(len(feat_t)):
                x_aligned = self.adaptation_layers[_i](feat_s[_i][b:b+1])
                # x_aligned = feat_s[_i][b:b+1]
                
                mask_01 = torch.full_like(mask_batch[b][_i], 1e-6)
                mask_01[mask_batch[b][_i] > 1e-6] = 1
                
                inverse_mask = torch.full_like(mask_batch[b][_i], 1e-6)
                inverse_mask[mask_batch[b][_i] == 1e-6] = 1
                inverse_mask[mask_batch[b][_i] > 1e-6] = 1e-6
                
                # feat_t_foreground[_i][b:b+1] = feat_t[_i][b:b+1] * mask_01
                # feat_t_background[_i][b:b+1] = feat_t[_i][b:b+1] * inverse_mask
            
                foreground_s = x_aligned * mask_01
                background_s = x_aligned * inverse_mask
                
                
                self.calculate_kd_loss_weight(feat_t[_i][b:b+1] * mask_01, foreground_s, _i, b, kd_feat_losses, 
                                       kd_channel_losses, kd_spatial_losses, cfg, mask_batch[b][_i])
                self.calculate_kd_loss(feat_t[_i][b:b+1] * inverse_mask, background_s, _i, b, kd_feat_losses, 
                                       kd_channel_losses, kd_spatial_losses, cfg)
                
                
            feat_loss_record += kd_feat_losses[b]
            channel_loss_record += kd_channel_losses[b]
            spatial_loss_record += kd_spatial_losses[b]
        
        losses.update({'kd_feat_loss': feat_loss_record})
        losses.update({'kd_relation_loss': realtion_loss_record})
        losses.update({'kd_channel_loss': channel_loss_record})
        # 要不要待定
        losses.update({'kd_spatial_loss': spatial_loss_record})
        
        return losses

    def loss_old(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             model_t = None,
             cfg = None) -> Union[dict, list]:

        if cfg.kd:
            batch_size = batch_inputs.shape[0]
            bboxes_per_image = [[] for _ in range(batch_size)]
            bboxes_labels = batch_data_samples['bboxes_labels']
            # 遍历所有边界框
            for bbox_label in bboxes_labels:
                img_idx = int(bbox_label[0].item())  # 获取图像索引
                bbox = bbox_label[2:]                # 获取边界框坐标 [x_min, y_min, x_max, y_max]
                
                # 将当前边界框添加到对应图像索引的列表中
                bboxes_per_image[img_idx].append(bbox.tolist())

            # 将每个图像的边界框列表转换为tensor
            gt_bboxes = [torch.tensor(bboxes) if bboxes else torch.empty((0, 4)) for bboxes in bboxes_per_image]
        
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        
        
        if cfg.kd:
            feat_t = model_t.extract_feat(batch_inputs)
            losses = self.caculate_loss(feat_t, x, losses, cfg, gt_bboxes, batch_inputs)
        
        return losses

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             model_t = None,
             cfg = None) -> Union[dict, list]:
        
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)

        if cfg.kd_new:
            # batch_size = batch_inputs.shape[0]
            # bboxes_per_image = [[] for _ in range(batch_size)]
            # bboxes_labels = batch_data_samples['bboxes_labels']
            # # 遍历所有边界框
            # for bbox_label in bboxes_labels:
            #     img_idx = int(bbox_label[0].item())  # 获取图像索引
            #     bbox = bbox_label[2:]                # 获取边界框坐标 [x_min, y_min, x_max, y_max]
                
            #     # 将当前边界框添加到对应图像索引的列表中
            #     bboxes_per_image[img_idx].append(bbox.tolist())

            # # 将每个图像的边界框列表转换为tensor
            # gt_bboxes = [torch.tensor(bboxes) if bboxes else torch.empty((0, 4)) for bboxes in bboxes_per_image]
            # feat_t = model_t.extract_feat(batch_inputs)
            # losses = self.caculate_loss(feat_t, x, losses, cfg, gt_bboxes, batch_inputs)
            
            feat_map_t = model_t.extract_feat(batch_inputs)
            backbone_feat_t = cfg.backbone_feat['feat_t']
            backbone_feat_s = self.backbone_adaptation(cfg.backbone_feat['feat_s'])
            criterion = nn.MSELoss()
            loss_backbone_feat = criterion(backbone_feat_s, backbone_feat_t) * cfg.weight_fitnet
            
            batch_size = batch_inputs.shape[0]
            bboxes_per_image = [[] for _ in range(batch_size)]
            bboxes_labels = batch_data_samples['bboxes_labels']
            # 遍历所有边界框
            for bbox_label in bboxes_labels:
                img_idx = int(bbox_label[0].item())  # 获取图像索引
                bbox = bbox_label[2:]                # 获取边界框坐标 [x_min, y_min, x_max, y_max]
                # 将当前边界框添加到对应图像索引的列表中
                bboxes_per_image[img_idx].append(bbox.tolist())
            gt_bboxes = [torch.tensor(bboxes) if bboxes else torch.empty((0, 4)) for bboxes in bboxes_per_image]
            _mask = self.mask_insdist(gt_bboxes, feat_map_t[-1], featmap_size=feat_map_t[-1].shape[2:], featmap_stride = 32, threshold=0.6).unsqueeze(1)
            feat_loss = 0.
            for i in range(0, len(feat_map_t)):
                d_size = feat_map_t[i].shape[2:]
                mask = F.interpolate(_mask, d_size).squeeze(1)
                loss_gt, loss_bg = self.dist_insdist(feat_map_t[i], self.adaptation_layers[i](x[i]), mask)
                feat_loss += (loss_gt * 1 + loss_bg * 1)
            feat_loss = feat_loss * cfg.weight_insdist
            
            
            feat_loss_record = 0.
            for i in range(len(feat_map_t)):
                feat_t = feat_map_t[i]
                feat_s = self.adaptation_layers[i](x[i])
                norm_S, norm_T = self.norm(feat_s), self.norm(feat_t)
                feat_loss_record += F.mse_loss(norm_S, norm_T) / 2
            feat_loss_record = feat_loss_record * cfg.weight_pkd  + loss_backbone_feat + feat_loss #  0.1
            losses.update({'kd_feat_loss': feat_loss_record})
            
            
            loss_cls = 0.
            stu_outputs = self.bbox_head(x)
            tea_outputs = model_t.bbox_head(feat_map_t)
            stu_cls = stu_outputs[0]
            tea_cls = tea_outputs[0]
            for stu, tea in zip(stu_cls, tea_cls):
                stu = stu.view(stu.shape[0],  int(stu.shape[1] / cfg.num_classes), cfg.num_classes, -1)
                tea = tea.view(tea.shape[0],  int(tea.shape[1] / cfg.num_classes), cfg.num_classes, -1)
                teacher_probs = F.softmax(tea / 4.0, dim=2)
                student_probs = F.log_softmax(stu / 4.0, dim=2)
                loss_cls += F.kl_div(student_probs, teacher_probs, reduction='mean') * (4.0 ** 2)
            loss_cls = loss_cls * cfg.weight_kdori
            losses.update({'kd_realation_loss': loss_cls})
                
        elif cfg.kd_ori:
            loss_cls = 0.
            stu_outputs = self.bbox_head(x)
            feat_map_t = model_t.extract_feat(batch_inputs)
            tea_outputs = model_t.bbox_head(feat_map_t)
            stu_cls = stu_outputs[0]
            tea_cls = tea_outputs[0]
            for stu, tea in zip(stu_cls, tea_cls):
                stu = stu.view(stu.shape[0],  int(stu.shape[1] / cfg.num_classes), cfg.num_classes, -1)
                tea = tea.view(tea.shape[0],  int(tea.shape[1] / cfg.num_classes), cfg.num_classes, -1)
                teacher_probs = F.softmax(tea / 4.0, dim=2)
                student_probs = F.log_softmax(stu / 4.0, dim=2)
                loss_cls += F.kl_div(student_probs, teacher_probs, reduction='mean') * (4.0 ** 2)
            loss_cls = loss_cls * cfg.weight_kdori
            losses.update({'kd_cls_loss': loss_cls})
            
        elif cfg.fitnet:
            # TODO：backbone hook
            feat_map_t = model_t.extract_feat(batch_inputs)
            backbone_feat_t = cfg.backbone_feat['feat_t']
            backbone_feat_s = self.backbone_adaptation(cfg.backbone_feat['feat_s'])
            criterion = nn.MSELoss()
            loss_backbone_feat = criterion(backbone_feat_s, backbone_feat_t) * cfg.weight_fitnet
            losses.update({'kd_backbone_feat_loss': loss_backbone_feat})
        
        elif cfg.insdist:
            feat_map_t = model_t.extract_feat(batch_inputs)
            batch_size = batch_inputs.shape[0]
            bboxes_per_image = [[] for _ in range(batch_size)]
            bboxes_labels = batch_data_samples['bboxes_labels']
            # 遍历所有边界框
            for bbox_label in bboxes_labels:
                img_idx = int(bbox_label[0].item())  # 获取图像索引
                bbox = bbox_label[2:]                # 获取边界框坐标 [x_min, y_min, x_max, y_max]
                # 将当前边界框添加到对应图像索引的列表中
                bboxes_per_image[img_idx].append(bbox.tolist())
            gt_bboxes = [torch.tensor(bboxes) if bboxes else torch.empty((0, 4)) for bboxes in bboxes_per_image]
            _mask = self.mask_insdist(gt_bboxes, feat_map_t[-1], featmap_size=feat_map_t[-1].shape[2:], featmap_stride = 32, threshold=0.6).unsqueeze(1)
            feat_loss = 0
            for i in range(0, len(feat_map_t)):
                d_size = feat_map_t[i].shape[2:]
                mask = F.interpolate(_mask, d_size).squeeze(1)
                loss_gt, loss_bg = self.dist_insdist(feat_map_t[i], self.adaptation_layers[i](x[i]), mask)
                feat_loss += (loss_gt * 1 + loss_bg * 1)
            feat_loss = feat_loss * cfg.weight_insdist
            # losses.update({'kd_feat_loss': feat_loss})
            
            feat_loss_record = torch.tensor(0.).cuda()
            for i in range(len(feat_map_t)):
                feat_t = feat_map_t[i]
                feat_s = self.adaptation_layers[i](x[i])
                norm_S, norm_T = self.norm(feat_s), self.norm(feat_t)
                feat_loss_record += F.mse_loss(norm_S, norm_T) / 2
            loss = feat_loss + feat_loss_record * 6
            losses.update({'kd_feat_loss': loss})
        
        elif cfg.arsd:
            stu_outputs = self.bbox_head(x)
            feat_map_t = model_t.extract_feat(batch_inputs)
            tea_outputs = model_t.bbox_head(feat_map_t)
            
            stu_cls = stu_outputs[0]
            tea_cls = tea_outputs[0]
            
            batch_size = batch_inputs.shape[0]
            bboxes_per_image = [[] for _ in range(batch_size)]
            bboxes_labels = batch_data_samples['bboxes_labels']
            # 遍历所有边界框
            for bbox_label in bboxes_labels:
                img_idx = int(bbox_label[0].item())  # 获取图像索引
                bbox = bbox_label[2:]                # 获取边界框坐标 [x_min, y_min, x_max, y_max]
                # 将当前边界框添加到对应图像索引的列表中
                bboxes_per_image[img_idx].append(bbox.tolist())
            gt_bboxes = [torch.tensor(bboxes) if bboxes else torch.empty((0, 4)) for bboxes in bboxes_per_image]
            
            stu_adapt = []
            for i in range(len(x)):
                stu_ada = []
                stu_ada = self.adaptation_layers[i](x[i])
                stu_adapt.append(stu_ada)
            
            mask_batch = self.get_masks_arsd(x, gt_bboxes, batch_inputs)
            
            # FPN
            sup_loss_fpn = 0.0
            norm = 0
            for j, mask_per_im in enumerate(mask_batch):
                for i, mask_per_level in enumerate(mask_per_im):
                    mask_per_level = (mask_per_level > 0).float().unsqueeze(0)
                    norm += mask_per_level.sum() * 2
                sup_loss_fpn += (torch.pow(stu_adapt[i][j] - feat_map_t[i][j], 2) * mask_per_level).sum()
            sup_loss_fpn = sup_loss_fpn * 0.01 / (norm + 1)
            sup_loss_fpn = sup_loss_fpn * cfg.weight_arsd_feat
            # losses.update({'kd_feat_loss': sup_loss_fpn})
            
            feat_loss_record = torch.tensor(0.).cuda()
            for i in range(len(feat_map_t)):
                feat_t = feat_map_t[i]
                feat_s = self.adaptation_layers[i](x[i])
                norm_S, norm_T = self.norm(feat_s), self.norm(feat_t)
                feat_loss_record += F.mse_loss(norm_S, norm_T) / 2
            loss = sup_loss_fpn + feat_loss_record * 8
            losses.update({'kd_feat_loss': loss})
            
            # 分类
            loss_cls = 0.
            for stu, tea in zip(stu_cls, tea_cls):
                stu = stu.view(stu.shape[0],  int(stu.shape[1] / cfg.num_classes), cfg.num_classes, -1)
                tea = tea.view(tea.shape[0],  int(tea.shape[1] / cfg.num_classes), cfg.num_classes, -1)
                teacher_probs = F.softmax(tea / 4.0, dim=2)
                student_probs = F.log_softmax(stu / 4.0, dim=2)
                loss_cls += F.kl_div(student_probs, teacher_probs, reduction='mean') * (4.0 ** 2)
            loss_cls = loss_cls * cfg.weight_arsd_cls
            losses.update({'kd_cls_loss': loss_cls})
        
        elif cfg.pkd:
            feat_loss_record = torch.tensor(0.).cuda()
            feat_map_t = model_t.extract_feat(batch_inputs)
            for i in range(len(feat_map_t)):
                feat_t = feat_map_t[i]
                feat_s = self.adaptation_layers[i](x[i])
                norm_S, norm_T = self.norm(feat_s), self.norm(feat_t)
                feat_loss_record += F.mse_loss(norm_S, norm_T) / 2
            feat_loss_record = feat_loss_record *  cfg.weight_pkd #  0.1
            losses.update({'kd_feat_loss': feat_loss_record})
            return losses
        
        elif cfg.cwd:
            feat_loss_record = torch.tensor(0.).cuda()
            feat_map_t = model_t.extract_feat(batch_inputs)
            for i in range(len(feat_map_t)):
                feat_t = feat_map_t[i]
                feat_s = self.adaptation_layers[i](x[i])
                N, C, W, H = feat_s.shape
                softmax_pred_T = F.softmax(feat_t.view(-1, W*H)/4, dim=1)
                logsoftmax = torch.nn.LogSoftmax(dim=1)
                feat_loss_record += torch.sum(- softmax_pred_T * logsoftmax(feat_s.view(-1, W*H)/4.)) * (4. ** 2)
            feat_loss_record = feat_loss_record * cfg.weight_cwd # 5e-7
            losses.update({'kd_feat_loss': feat_loss_record})
            return losses
        
        elif cfg.defeat:
            feat_loss_record = torch.tensor(0.).cuda()
            feat_map_t = model_t.extract_feat(batch_inputs)
            
            batch_size = batch_inputs.shape[0]
            bboxes_per_image = [[] for _ in range(batch_size)]
            bboxes_labels = batch_data_samples['bboxes_labels']
            # 遍历所有边界框
            for bbox_label in bboxes_labels:
                img_idx = int(bbox_label[0].item())  # 获取图像索引
                bbox = bbox_label[2:]                # 获取边界框坐标 [x_min, y_min, x_max, y_max]
                # 将当前边界框添加到对应图像索引的列表中
                bboxes_per_image[img_idx].append(bbox.tolist())
            gt_bboxes = [torch.tensor(bboxes) if bboxes else torch.empty((0, 4)) for bboxes in bboxes_per_image]
            
            mask_batch = self.get_masks_01(x, gt_bboxes, batch_inputs)
            
            # feat_map_t_foreground = copy.deepcopy(feat_map_t)
            # feat_map_t_background = copy.deepcopy(feat_map_t)
            for b in range(batch_size):
                for i in range(len(feat_map_t)):
                    # 对齐学生特征图的尺寸
                    x_aligned = self.adaptation_layers[i](x[i][b:b+1])
                    
                    mask_01 = torch.where(mask_batch[b][i] > 1e-10, torch.tensor(1.).cuda(), torch.full_like(mask_batch[b][i], 1e-10))
                    inverse_mask = torch.where(mask_batch[b][i] > 1e-10, torch.full_like(mask_batch[b][i], 1e-10), torch.tensor(1.).cuda())
                    
                    foreground_feat_map_t = feat_map_t[i][b:b+1] * mask_01
                    background_feat_map_t = feat_map_t[i][b:b+1] * inverse_mask
                
                    foreground_s = x_aligned * mask_01
                    background_s = x_aligned * inverse_mask
                    
                    feat_loss_record += F.mse_loss(foreground_feat_map_t, foreground_s)
                    feat_loss_record += F.mse_loss(background_feat_map_t, background_s)
            feat_loss_record = feat_loss_record * cfg.weight_defeat   # 0.005
            losses.update({'kd_feat_loss': feat_loss_record})
        
        
        return losses

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-10)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def dist_insdist(self, tensor_a, tensor_b, mask):
        diff = (tensor_a - tensor_b) ** 2
    
        mask_gt = mask.unsqueeze(1).repeat(1, tensor_a.size(1), 1, 1).cuda()
        diff_gt = diff * mask_gt
        diff_gt = (torch.sum(diff_gt) + 1e-8) ** 0.5
        
        mask_bg = (1 - mask_gt)
        diff_bg = diff * mask_bg
        diff_bg = (torch.sum(diff_bg) + 1e-8) ** 0.5
        
        return diff_gt, diff_bg

    def mask_insdist(self, gt_bboxes, backbone_feat, featmap_size, featmap_stride, threshold):
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        with torch.no_grad():
            mask_batch = []
            for batch in range(len(gt_bboxes)):
                
                h, w = featmap_size[0], featmap_size[1]
                mask_per_img = torch.zeros([h, w], dtype=torch.double).cuda()
                
                for ins in range(gt_bboxes[batch].shape[0]):
                    gt_level_map = gt_bboxes[batch][ins] / featmap_stride
                    
                    lx = min(max(0, int(gt_level_map[0])), w - 1)
                    rx = min(max(0, int(gt_level_map[2])), w - 1)
                    ly = min(max(0, int(gt_level_map[1])), h - 1)
                    ry = min(max(0, int(gt_level_map[3])), h - 1)
                    
                    if (lx == rx) or (ly == ry):
                        mask_per_img[ly, lx] += 1
                    else:
                        x = backbone_feat[batch].view(-1, h * w).permute(1, 0)
                        feature_gt = avgpool(backbone_feat[batch][:, ly:(ry + 1), lx:(rx + 1)]).squeeze(-1)
                        energy = torch.mm(x, feature_gt)
                        
                        min_ = torch.min(energy)
                        max_ = torch.max(energy)
                        assert max_ != 0 
                        energy = (energy - min_) / max_
                        attention = energy.view(h, w)
                        
                        attention = (attention > threshold).double()
                        mask_per_img += attention
                mask_per_img = (mask_per_img > 0).double()
                mask_batch.append(mask_per_img)
                
        return torch.stack(mask_batch, dim=0)

    def get_masks_01(self, feat_maps, gt_bboxes, img):
        # 获取每个特征图的尺寸（高和宽）
        featmap_sizes = [featmap.size()[-2:] for featmap in feat_maps]
        # 获取原始图像的尺寸
        img_ori = img.size()[-2:]
        img_ori = torch.tensor([img_ori])   # 接近 [600, 1000]
        # 初始化一个列表，用于存储每个批次图像的掩码
        mask_batch = []

        # 获取批次大小
        batch_size = len(gt_bboxes)
        # 获取特征图的层级数
        levels = len(feat_maps)
        # 遍历每个图像
        for i in range(batch_size):
            # 初始化一个列表，用于存储每个图像的掩码
            mask_per_im = []
            # 计算真实边界框的面积
            area = torch.sqrt((gt_bboxes[i][:, 2] - gt_bboxes[i][:, 0]) * (gt_bboxes[i][:, 3] - gt_bboxes[i][:, 1]))
            # area_avg = torch.sum(area) / area.size(0)
                
            # 计算原始图像的面积
            area_image = torch.sqrt((img_ori[:, 0] * img_ori[:, 1]).to(torch.float32)).cuda()
            # 确定每个边界框应该分配给哪个特征图层级
            target_lvls = torch.floor(4 + torch.log2(area.cuda() / area_image + 1e-10)).cuda()
            # 将层级限制在0到4之间
            target_lvls = target_lvls.clamp(min=0, max=4).long()
            # 遍历每个特征图层级
            for j in range(levels):
                # 获取当前层级的高和宽
                height, width = featmap_sizes[j][0], featmap_sizes[j][1]
                # 初始化当前层级的掩码，初始值为0
                mask_per_level = torch.zeros([height, width], dtype=torch.float32).cuda()
                # 获取当前特征图的尺寸
                fea_scale = featmap_sizes[j]
                # 将特征图尺寸转换为tensor
                fea_scale = torch.tensor([fea_scale])
                # 计算特征图相对于原始图像的缩放比例
                percent = torch.div(fea_scale.float(), img_ori.float()).cuda()
                # 初始化一个用于存放缩放后的边界框的tensor
                gt_trans = torch.zeros(gt_bboxes[i].shape, dtype=torch.float32).cuda()
                # 将边界框按比例缩放到对应的特征图尺寸
                gt_trans[:, :2] = gt_bboxes[i][:, :2].cuda() * percent
                gt_trans[:, 2:] = gt_bboxes[i][:, 2:].cuda() * percent
                # 初始化一个与边界框尺寸相同的掩码
                gt_mask = torch.zeros(gt_bboxes[i].shape, dtype=torch.float32).cuda()
                # 为当前层级的每个边界框生成掩码
                for k in range(gt_bboxes[i].shape[0]):
                    # 如果边界框分配给当前层级，则将掩码设置为1
                    if target_lvls[k] == j:
                        gt_mask[k, :] = 1
                # 将掩码和缩放后的边界框相乘，得到当前层级的边界框
                gt_trans = gt_trans * gt_mask
                # 遍历每个边界框
                for k in range(gt_bboxes[i].shape[0]):
                    # 如果边界框为空，则跳过
                    if torch.sum(gt_bboxes[i][k]) == 0.:
                        break
                    # 为每个边界框生成特征图尺寸的掩码
                    mask_per_gt = torch.zeros(height, width, dtype=torch.float32).cuda()
                    # 将掩码中边界框所在位置的值设置为对应的加权权重
                    mask_per_gt[gt_trans[k][1].int():gt_trans[k][3].int(), gt_trans[k][0].int():gt_trans[k][2].int()] = 1
                    # 将当前边界框的掩码叠加到当前层级的掩码上
                    mask_per_level += mask_per_gt
                # 将当前层级的掩码添加到图像的掩码列表中
                mask_per_im.append(mask_per_level)
            # 将图像的掩码列表添加到批次的掩码列表中
            mask_batch.append(mask_per_im)

        return mask_batch

    def get_masks_arsd(self, feat_maps, gt_bboxes, img):
        # 获取每个特征图的尺寸（高和宽）
        featmap_sizes = [featmap.size()[-2:] for featmap in feat_maps] # [torch.Size([76, 128]), torch.Size([38, 64]), torch.Size([19, 32]), torch.Size([10, 16]), torch.Size([5, 8])]
        # 获取设备信息，通常是GPU
        # device = feat_maps[0].device
        # 获取原始图像的尺寸
        img_ori = img.size()[-2:]
        img_ori = torch.tensor([img_ori])   # 接近 [600, 1000]
        # 获取锚点（anchors）和有效标志（valid flags）列表
        # anchor_list, valid_flag_list = self.bbox_head.get_anchors(featmap_sizes, img_metas, device=device)
        # 初始化一个列表，用于存储每个批次图像的掩码
        mask_batch = []

        for bbox in gt_bboxes:
            bbox = bbox.cuda()
        # 获取批次大小
        batch_size = len(gt_bboxes)
        # 获取特征图的层级数
        levels = len(feat_maps)
        # 遍历每个图像
        for i in range(batch_size):
            # 初始化一个列表，用于存储每个图像的掩码
            mask_per_im = []
            # 计算真实边界框的面积
            area = torch.sqrt((gt_bboxes[i][:, 2] - gt_bboxes[i][:, 0]) * (gt_bboxes[i][:, 3] - gt_bboxes[i][:, 1]))
            area = area.cuda()
            # 计算原始图像的面积
            area_image = torch.sqrt((img_ori[:, 0] * img_ori[:, 1]).to(torch.float32)).cuda()
            # 确定每个边界框应该分配给哪个特征图层级
            target_lvls = torch.floor(4 + torch.log2(area / area_image + 1e-10)).cuda()
            # 将层级限制在0到4之间
            target_lvls = target_lvls.clamp(min=0, max=4).long()

            # 遍历每个特征图层级
            for j in range(levels):
                # 获取当前层级的高和宽
                height, width = featmap_sizes[j][0], featmap_sizes[j][1]
                # 初始化当前层级的掩码，初始值为0
                mask_per_level = torch.zeros([height, width], dtype=torch.float32).cuda()
                # 获取当前特征图的尺寸
                fea_scale = featmap_sizes[j]
                # 将特征图尺寸转换为tensor
                fea_scale = torch.tensor([fea_scale])
                # 计算特征图相对于原始图像的缩放比例
                percent = torch.div(fea_scale.float(), img_ori.float()).cuda()
                # 初始化一个用于存放缩放后的边界框的tensor
                gt_trans = torch.zeros(gt_bboxes[i].shape, dtype=torch.float32).cuda()
                # 将边界框按比例缩放到对应的特征图尺寸
                gt_trans[:, :2] = gt_bboxes[i][:, :2].cuda() * percent
                gt_trans[:, 2:] = gt_bboxes[i][:, 2:].cuda() * percent
                # 初始化一个与边界框尺寸相同的掩码
                gt_mask = torch.zeros(gt_bboxes[i].shape, dtype=torch.float32).cuda()
                # 为当前层级的每个边界框生成掩码
                for k in range(gt_bboxes[i].shape[0]):
                    # 如果边界框分配给当前层级，则将掩码设置为1
                    if target_lvls[k] == j:
                        gt_mask[k, :] = 1
                # 将掩码和缩放后的边界框相乘，得到当前层级的边界框
                gt_trans = gt_trans * gt_mask
                # 遍历每个边界框
                for k in range(gt_bboxes[i].shape[0]):
                    # 如果边界框为空，则跳过
                    if torch.sum(gt_bboxes[i][k]) == 0.:
                        break
                    # 为每个边界框生成特征图尺寸的掩码
                    mask_per_gt = torch.zeros(height, width, dtype=torch.float32).cuda()
                    # 将掩码中边界框所在位置的值设置为对应的加权权重
                    mask_per_gt[gt_trans[k][1].int():gt_trans[k][3].int(), gt_trans[k][0].int():gt_trans[k][2].int()] = 1.0
                    # 将当前边界框的掩码叠加到当前层级的掩码上
                    mask_per_level += mask_per_gt
                epsilon = 1e-10  # 一个小的常数，用于平滑掩码防止梯度消失
                mask_per_level = mask_per_level.float()
                mask_per_level = epsilon + (1 - epsilon) * mask_per_level
                # 将当前层级的掩码添加到图像的掩码列表中
                mask_per_im.append(mask_per_level)
            # 将图像的掩码列表添加到批次的掩码列表中
            mask_batch.append(mask_per_im)

        return mask_batch

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x



@MODELS.register_module()
class SingleStageDetector_quanti_fcos_2(BaseDetector_quanti_fcos_2):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x


@MODELS.register_module()
class SingleStageDetector_quanti(BaseDetector_quanti):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x



@MODELS.register_module()
class SingleStageDetector_quanti_fcos(BaseDetector_quanti_fcos):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x



@MODELS.register_module()
class SingleStageDetector_quanti_retinanet(BaseDetector_quanti_retinanet):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x
