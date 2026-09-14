# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector, SingleStageDetector_quanti_fcos, SingleStageDetector_quanti_fcos_2, SingleStageDetector_quanti_retinanet


@MODELS.register_module()
class FCOS(SingleStageDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of FCOS. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of FCOS. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)


# @MODELS.register_module()
# class FCOS_quanti_2(SingleStageDetector_quanti_fcos_2):
#     """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_

#     Args:
#         backbone (:obj:`ConfigDict` or dict): The backbone config.
#         neck (:obj:`ConfigDict` or dict): The neck config.
#         bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
#         train_cfg (:obj:`ConfigDict` or dict, optional): The training config
#             of FCOS. Defaults to None.
#         test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
#             of FCOS. Defaults to None.
#         data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
#             :class:`DetDataPreprocessor` to process the input data.
#             Defaults to None.
#         init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
#             list[dict], optional): Initialization config dict.
#             Defaults to None.
#     """

#     def __init__(self,
#                  backbone: ConfigType,
#                  neck: ConfigType,
#                  bbox_head: ConfigType,
#                  train_cfg: OptConfigType = None,
#                  test_cfg: OptConfigType = None,
#                  data_preprocessor: OptConfigType = None,
#                  init_cfg: OptMultiConfig = None) -> None:
#         super().__init__(
#             backbone=backbone,
#             neck=neck,
#             bbox_head=bbox_head,
#             train_cfg=train_cfg,
#             test_cfg=test_cfg,
#             data_preprocessor=data_preprocessor,
#             init_cfg=init_cfg)


@MODELS.register_module()
class FCOS_quanti(SingleStageDetector_quanti_retinanet):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of FCOS. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of FCOS. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
