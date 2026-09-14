# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Tuple, Union

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector, BaseDetector_quanti_frcnn
from typing import List, Optional, Tuple, Union
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import delta2bbox
from mmengine.structures import InstanceData
from mmdet.structures.bbox import (cat_boxes, empty_box_as, get_box_tensor,
                                   get_box_wh, scale_boxes)
from mmcv.ops import batched_nms
# from mmcv.ops.nms import batched_nms

torch.fx.wrap("scale_boxes")
torch.fx.wrap("get_box_wh")
torch.fx.wrap("empty_box_as")
torch.fx.wrap("batched_nms")
torch.fx.wrap("cat_boxes")
torch.fx.wrap("get_box_tensor")
torch.fx.wrap("batched_nms")

# @torch.fx.wrap
# def batched_nms(boxes: Tensor,
#                 scores: Tensor,
#                 idxs: Tensor,
#                 nms_cfg,
#                 class_agnostic: bool = False) -> Tuple[Tensor, Tensor]:
    
#     # skip nms when nms_cfg is None
#     if nms_cfg is None:
#         scores, inds = scores.sort(descending=True)
#         boxes = boxes[inds]
#         return torch.cat([boxes, scores[:, None]], -1), inds

#     nms_cfg_ = nms_cfg.copy()
#     class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
#     if class_agnostic:
#         boxes_for_nms = boxes
#     else:
#         # When using rotated boxes, only apply offsets on center.
#         if boxes.size(-1) == 5:
#             # Strictly, the maximum coordinates of the rotating box
#             # (x,y,w,h,a) should be calculated by polygon coordinates.
#             # But the conversion from rotated box to polygon will
#             # slow down the speed.
#             # So we use max(x,y) + max(w,h) as max coordinate
#             # which is larger than polygon max coordinate
#             # max(x1, y1, x2, y2,x3, y3, x4, y4)
#             max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
#             offsets = idxs.to(boxes) * (
#                 max_coordinate + torch.tensor(1).to(boxes))
#             boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
#             boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 2:5]],
#                                       dim=-1)
#         else:
#             max_coordinate = boxes.max()
#             offsets = idxs.to(boxes) * (
#                 max_coordinate + torch.tensor(1).to(boxes))
#             boxes_for_nms = boxes + offsets[:, None]

#     nms_type = nms_cfg_.pop('type', 'nms')
#     nms_op = eval(nms_type)

#     split_thr = nms_cfg_.pop('split_thr', 10000)
#     # Won't split to multiple nms nodes when exporting to onnx
#     if boxes_for_nms.shape[0] < split_thr:
#         dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
#         boxes = boxes[keep]

#         # This assumes `dets` has arbitrary dimensions where
#         # the last dimension is score.
#         # Currently it supports bounding boxes [x1, y1, x2, y2, score] or
#         # rotated boxes [cx, cy, w, h, angle_radian, score].

#         scores = dets[:, -1]
#     else:
#         max_num = nms_cfg_.pop('max_num', -1)
#         total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
#         # Some type of nms would reweight the score, such as SoftNMS
#         scores_after_nms = scores.new_zeros(scores.size())
#         for id in torch.unique(idxs):
#             mask = (idxs == id).nonzero(as_tuple=False).view(-1)
#             dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
#             total_mask[mask[keep]] = True
#             scores_after_nms[mask[keep]] = dets[:, -1]
#         keep = total_mask.nonzero(as_tuple=False).view(-1)

#         scores, inds = scores_after_nms[keep].sort(descending=True)
#         keep = keep[inds]
#         boxes = boxes[keep]

#         if max_num > 0:
#             keep = keep[:max_num]
#             boxes = boxes[:max_num]
#             scores = scores[:max_num]

#     boxes = torch.cat([boxes, scores[:, None]], -1)
#     return boxes, keep

@torch.fx.wrap
def unpack_img_metas(batch_data_samples: SampleList) -> List:
    """Wrap this function with `torch.fx.wrap` so that this function can be
    regarded as a leaf node in the graph traced by `torch.fx`."""
    batch_img_metas = [
        data_samples.metainfo for data_samples in batch_data_samples
    ]
    return batch_img_metas

@torch.fx.wrap
def bbox2roi(bbox_list) -> Tensor:
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        bboxes = get_box_tensor(bboxes)
        img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
        rois = torch.cat([img_inds, bboxes], dim=-1)
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois

@torch.fx.wrap
def generate_proposal(rpn_results_list):
    proposal = [rpn_results.bboxes for rpn_results in rpn_results_list]
    return proposal

@torch.fx.wrap
def select_single_mlvl(mlvl_tensors, batch_id, detach=True):
    assert isinstance(mlvl_tensors, (list, tuple))
    num_levels = len(mlvl_tensors)

    if detach:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id].detach() for i in range(num_levels)
        ]
    else:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id] for i in range(num_levels)
        ]
    return mlvl_tensor_list

@torch.fx.wrap
def gen_base_anchors() -> List[Tensor]:
    """Generate base anchors.

    Returns:
        list(torch.Tensor): Base anchors of a feature grid in multiple \
            feature levels.
    """
    base_sizes = [4, 8, 16, 32, 64]
    multi_level_base_anchors = []
    for i, base_size in enumerate(base_sizes):
        center = None
        multi_level_base_anchors.append(
                                        gen_single_level_base_anchors(
                                        base_size,
                                        scales=torch.Tensor([8]),
                                        ratios=torch.Tensor([0.5, 1.0, 2.0]),
                                        center=center))
    return multi_level_base_anchors

@torch.fx.wrap
def gen_single_level_base_anchors(
                                base_size: Union[int, float],
                                scales: Tensor,
                                ratios: Tensor,
                                center: Optional[Tuple[float]] = None) -> Tensor:
    center_offset = 0.0
    w = base_size
    h = base_size
    if center is None:
        x_center = center_offset * w
        y_center = center_offset * h
    else:
        x_center, y_center = center

    h_ratios = torch.sqrt(ratios)
    w_ratios = 1 / h_ratios
    
    ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)

    base_anchors = [
        x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
        y_center + 0.5 * hs
    ]
    base_anchors = torch.stack(base_anchors, dim=-1)

    return base_anchors

@torch.fx.wrap
def _meshgrid(
            x: Tensor,
            y: Tensor,
            row_major: bool = True) -> Tuple[Tensor]:

    # use shape instead of len to keep tracing while exporting to onnx
    xx = x.repeat(y.shape[0])
    yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx

@torch.fx.wrap
def grid_priors(featmap_sizes: List[Tuple],
                dtype: torch.dtype = torch.float32,
                device = 'cuda') -> List[Tensor]:
    
    multi_level_anchors = []
    for i in range(5):
        anchors = single_level_grid_priors(
            featmap_sizes[i], level_idx=i, dtype=dtype, device=device)
        multi_level_anchors.append(anchors)
    return multi_level_anchors

@torch.fx.wrap
def single_level_grid_priors(featmap_size: Tuple[int, int],
                            level_idx: int,
                            dtype: torch.dtype = torch.float32,
                            device = 'cuda') -> Tensor:
    base_anchors = gen_base_anchors()
    base_anchors = base_anchors[level_idx].to(device).to(dtype)
    feat_h, feat_w = featmap_size
    strides = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]
    stride_w, stride_h = strides[level_idx]
    # First create Range with the default dtype, than convert to
    # target `dtype` for onnx exporting.
    
    shift_x = torch.arange(0, feat_w, device=device).to(dtype) * stride_w
    shift_y = torch.arange(0, feat_h, device=device).to(dtype) * stride_h
    
    # shift_x = custom_arange(feat_w, device=device, dtype=dtype) * stride_w
    # shift_y = custom_arange(feat_h, device=device, dtype=dtype) * stride_w

    shift_xx, shift_yy = _meshgrid(shift_x, shift_y)
    shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
    # first feat_w elements correspond to the first row of shifts
    # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
    # shifted anchors (K, A, 4), reshape to (K*A, 4)

    all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
    all_anchors = all_anchors.view(-1, 4)
    # first A rows correspond to A anchors of (0, 0) in feature map,
    # then (0, 1), (0, 2), ...
    
    # if self.use_box_type:
    #     all_anchors = HorizontalBoxes(all_anchors)
    
    return all_anchors

@torch.fx.wrap
def decode(
        bboxes,
        pred_bboxes: Tensor,
        max_shape = None,
        wh_ratio_clip: Optional[float] = 16 / 1000
        ):
    
        bboxes = get_box_tensor(bboxes)
        assert pred_bboxes.size(0) == bboxes.size(0)
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1)

        if pred_bboxes.ndim == 2 and not torch.onnx.is_in_onnx_export():
            # single image decode
            means = [0.0, 0.0, 0.0, 0.0]
            stds = [1.0, 1.0, 1.0, 1.0]
            clip_border = True
            add_ctr_clamp = False
            ctr_clamp = 32
            decoded_bboxes = delta2bbox(bboxes, pred_bboxes, means,
                                        stds, max_shape, wh_ratio_clip,
                                        clip_border, add_ctr_clamp,
                                        ctr_clamp)
        else:
            if pred_bboxes.ndim == 3 and not torch.onnx.is_in_onnx_export():
                pass
            # decoded_bboxes = onnx_delta2bbox(bboxes, pred_bboxes, self.means,
            #                                  self.stds, max_shape,
            #                                  wh_ratio_clip, self.clip_border,
            #                                  self.add_ctr_clamp,
            #                                  self.ctr_clamp)

        return decoded_bboxes

@torch.fx.wrap
def _bbox_post_process(
                        results: InstanceData,
                        cfg,
                        rescale: bool = False,
                        with_nms: bool = True,
                        img_meta: Optional[dict] = None) -> InstanceData:
    assert with_nms, '`with_nms` must be True in RPNHead'
    if rescale:
        assert img_meta.get('scale_factor') is not None
        scale_factor = [1 / s for s in img_meta['scale_factor']]
        results.bboxes = scale_boxes(results.bboxes, scale_factor)

    # filter small size bboxes
    if cfg.get('min_bbox_size', -1) >= 0:
        w, h = get_box_wh(results.bboxes)
        valid_mask = (w > cfg.get('min_bbox_size')) & (h > cfg.get('min_bbox_size'))
        if not valid_mask.all():
            results = results[valid_mask]

    if results.bboxes.numel() > 0:
        bboxes = get_box_tensor(results.bboxes)
        det_bboxes, keep_idxs = batched_nms(bboxes, results.scores,
                                            results.level_ids, cfg.get('nms'))
        results = results[keep_idxs]
        # some nms would reweight the score, such as softnms
        results.scores = det_bboxes[:, -1]
        results = results[:cfg.get('max_per_img')]
        # TODO: This would unreasonably show the 0th class label
        #  in visualization
        results.labels = results.scores.new_zeros(
            len(results), dtype=torch.long)
        del results.level_ids
    else:
        # To avoid some potential error
        results_ = InstanceData()
        results_.bboxes = empty_box_as(results.bboxes)
        results_.scores = results.scores.new_zeros(0)
        results_.labels = results.scores.new_zeros(0)
        results = results_
    return results


@torch.fx.wrap
def _predict_by_feat_single(
                            cls_score_list: List[Tensor],
                            bbox_pred_list: List[Tensor],
                            score_factor_list: List[Tensor],
                            mlvl_priors: List[Tensor],
                            img_meta: dict,
                            cfg,
                            rescale: bool = False,
                            with_nms: bool = True):
    """
        iou_threshold 需要根据数据集修改
    """
    cfg = {'nms_pre': 1000, 'max_per_img': 1000, 'nms': {'type': 'nms', 'iou_threshold': 0.4}, 'min_bbox_size': 0}
    cfg = copy.deepcopy(cfg)
    img_shape = img_meta['img_shape']
    nms_pre = cfg.get('nms_pre', -1)

    mlvl_bbox_preds = []
    mlvl_valid_priors = []
    mlvl_scores = []
    level_ids = []
    for level_idx, (cls_score, bbox_pred, priors) in \
            enumerate(zip(cls_score_list, bbox_pred_list,
                            mlvl_priors)):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        encode_size = 4
        reg_dim = encode_size
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, reg_dim)
        cls_score = cls_score.permute(1, 2,
                                        0).reshape(-1, 1)
        scores = cls_score.sigmoid()

        scores = torch.squeeze(scores)
        
        if  0 < nms_pre < scores.shape[0]:
            # sort is faster than topk
            # _, topk_inds = scores.topk(cfg.nms_pre)
            ranked_scores, rank_inds = scores.sort(descending=True)
            topk_inds = rank_inds[:nms_pre]
            scores = ranked_scores[:nms_pre]
            bbox_pred = bbox_pred[topk_inds, :]
            priors = priors[topk_inds]

        mlvl_bbox_preds.append(bbox_pred)
        mlvl_valid_priors.append(priors)
        mlvl_scores.append(scores)

        # use level id to implement the separate level nms
        level_ids.append(
            scores.new_full((scores.size(0), ),
                            level_idx,
                            dtype=torch.long))

    bbox_pred = torch.cat(mlvl_bbox_preds)
    priors = cat_boxes(mlvl_valid_priors)
    bboxes = decode(priors, bbox_pred, max_shape=img_shape)

    
    results = InstanceData()
    results.bboxes = bboxes
    results.scores = torch.cat(mlvl_scores)
    results.level_ids = torch.cat(level_ids)

    return _bbox_post_process(
        results=results, cfg=cfg, rescale=rescale, img_meta=img_meta)


@torch.fx.wrap
def predict_by_feat(
                    cls_scores: List[Tensor],
                    bbox_preds: List[Tensor],
                    score_factors = None,
                    batch_img_metas = None,
                    cfg = None,
                    rescale: bool = False,
                    with_nms: bool = True):
    
    with_score_factors = False
    num_levels = len(cls_scores)

    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    mlvl_priors = grid_priors(
        featmap_sizes,
        dtype=cls_scores[0].dtype,
        device=cls_scores[0].device)

    result_list = []
    num_images = len(batch_img_metas)
    for img_id in range(num_images):
        img_meta = batch_img_metas[img_id]
        cls_score_list = select_single_mlvl(
            cls_scores, img_id, detach=True)
        bbox_pred_list = select_single_mlvl(
            bbox_preds, img_id, detach=True)
        if with_score_factors:
            score_factor_list = select_single_mlvl(
                score_factors, img_id, detach=True)
        else:
            score_factor_list = [None for _ in range(num_levels)]

        results = _predict_by_feat_single(
            cls_score_list=cls_score_list,
            bbox_pred_list=bbox_pred_list,
            score_factor_list=score_factor_list,
            mlvl_priors=mlvl_priors,
            img_meta=img_meta,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms)
        result_list.append(results)
    return result_list

@MODELS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading single-stage
        weights into two-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) != 0 and len(rpn_head_keys) == 0:
            for bbox_head_key in bbox_head_keys:
                rpn_head_key = rpn_head_prefix + \
                               bbox_head_key[len(bbox_head_prefix):]
                state_dict[rpn_head_key] = state_dict.pop(bbox_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

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

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        x = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_outs = self.roi_head.forward(x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs, )
        return results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

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
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples


@MODELS.register_module()
class TwoStageDetector_quanti(BaseDetector_quanti_frcnn):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading single-stage
        weights into two-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) != 0 and len(rpn_head_keys) == 0:
            for bbox_head_key in bbox_head_keys:
                rpn_head_key = rpn_head_prefix + \
                               bbox_head_key[len(bbox_head_prefix):]
                state_dict[rpn_head_key] = state_dict.pop(bbox_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

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

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        batch_img_metas = unpack_img_metas(batch_data_samples)
        
        x = self.extract_feat(batch_inputs)
        rpn_out = self.rpn_head(x)
        rpn_results_list = predict_by_feat(cls_scores=rpn_out[0], bbox_preds=rpn_out[1],
                                           batch_img_metas=batch_img_metas, rescale=False)
        
        proposals = generate_proposal(rpn_results_list)
        rois = bbox2roi(proposals)
        
        bbox_feats = self.roi_head.bbox_roi_extractor(x[:self.roi_head.bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred = self.roi_head.bbox_head(bbox_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        
        roi_outs = results + (bbox_results['cls_score'], bbox_results['bbox_pred'])
        
        return roi_outs
    
        # rpn_results_list = self.rpn_head.predict(x, batch_data_samples, rescale=False)
        
        # if self.with_rpn:
        #     rpn_results_list = self.rpn_head.predict(
        #         x, batch_data_samples, rescale=False)
        # else:
        #     assert batch_data_samples[0].get('proposals', None) is not None
        #     rpn_results_list = [
        #         data_sample.proposals for data_sample in batch_data_samples
        #     ]
        # roi_outs = self.roi_head.forward(x, rpn_results_list,
        #                                  batch_data_samples)
        results = results + (roi_outs, )
        return results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

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
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
