# Copyright (c) OpenMMLab. All rights reserved.
"""MMEngine training adapter around an MQBench FX detection graph."""

from typing import Optional

import torch
from mmengine.model import BaseModel

from mmdet.models.dense_heads.base_dense_head import unpack_img_metas
from mmdet.models.detectors.base import add_pred_to_datasample
from mmdet.models.utils import unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import SampleList


@MODELS.register_module()
class QATSingleStageGraphModel(BaseModel):
    """Restore loss/predict modes around an MQBench tensor-only FX graph."""

    def __init__(self, graph: torch.nn.Module,
                 data_preprocessor: torch.nn.Module):
        super().__init__(data_preprocessor=data_preprocessor)
        self.graph = graph

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[SampleList] = None,
                mode: str = 'tensor'):
        outputs = self.graph(inputs)
        if mode == 'tensor':
            return outputs
        if data_samples is None:
            raise ValueError(f'data_samples is required in {mode!r} mode')
        if mode == 'loss':
            batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = \
                unpack_gt_instances(data_samples)
            loss_inputs = outputs + (batch_gt_instances, batch_img_metas,
                                     batch_gt_instances_ignore)
            return self.graph.bbox_head.loss_by_feat(*loss_inputs)
        if mode == 'predict':
            predictions = self.graph.bbox_head.predict_by_feat(
                *outputs,
                batch_img_metas=unpack_img_metas(data_samples),
                rescale=True)
            return add_pred_to_datasample(data_samples, predictions)
        raise RuntimeError(f'Unsupported forward mode: {mode}')
