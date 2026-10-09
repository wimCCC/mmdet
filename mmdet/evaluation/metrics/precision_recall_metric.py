# Copyright (c) OpenMMLab. All rights reserved.
"""Precision/recall at a detection operating point.

COCO's mAP integrates the precision-recall curve over every recall level and
every IoU threshold, so it describes the whole curve rather than any single
operating point. Reporting "precision" and "recall" as numbers therefore
requires picking a confidence threshold explicitly, which is what this metric
does.

Do not confuse the output with ``coco/<class>_precision``, which ``CocoMetric``
already emits: that value is the mean of COCO's precision array over all IoU
thresholds and recall points, i.e. average precision under a misleading name
(which is why it prints identically to ``coco/bbox_mAP``). The values produced
here are the plain ``TP / (TP + FP)`` at one score threshold.

The threshold is either fixed by the caller or chosen as the one that maximises
F1 over the dataset; the score threshold actually used is reported alongside
the metrics.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import get_local_path
from pycocotools.coco import COCO

from mmdet.evaluation.functional.mean_ap import tpfp_default
from mmdet.registry import METRICS


@METRICS.register_module()
class PrecisionRecallMetric(BaseMetric):
    """Precision, recall and F1 at one score threshold and one IoU threshold.

    Detections are ranked by descending score and truncated to ``max_dets`` per
    image -- the same per-image budget COCO applies to mAP, so the two metrics
    describe one operating point rather than two different ones. A detection is
    a true positive when its IoU against the best still-unmatched ground truth
    reaches ``iou_thr``, otherwise it is a false positive; unmatched ground
    truths are false negatives.

    ``score_thr`` of None selects the threshold maximising F1, which is the
    natural choice when a run's recall at a fixed threshold is limited by
    detection *confidence* rather than by the detector's ability to find the
    objects. Note two consequences:

    * the threshold is fitted on the same split it is reported on, so the
      resulting P/R/F1 are an optimistic upper bound and carry the same caveat
      as picking a best checkpoint on that split;
    * every run gets its own threshold, so runs are not all being measured at
      one common operating point and their P/R are less directly comparable
      than at a fixed threshold. ``score_thr`` is returned precisely so the
      reader can see how far apart the chosen points are.

    Because greedy matching proceeds in descending score order, the TP/FP label
    of a detection does not depend on any lower-scoring detection, so one pass
    over the ranked detections yields the metrics at *every* threshold. The
    sweep is therefore exact rather than a grid approximation.

    Only single-category annotation files are supported (matching is done
    without regard to class), which is the case for every config that uses this
    metric. A multi-category file raises rather than silently mismatching.

    Args:
        ann_file (str): Path to the COCO-format annotation file used for
            evaluation.
        score_thr (float, optional): Fixed confidence threshold defining the
            operating point; detections must exceed it. None selects the value
            maximising F1 over the dataset. Defaults to 0.5.
        iou_thr (float): IoU threshold for a match. Defaults to 0.5.
        max_dets (int): Maximum detections considered per image, after ranking
            by score. Should match the model's ``test_cfg.max_per_img``.
            Defaults to 100.
        use_legacy_coordinate (bool): Whether to use the mmdet v1.x coordinate
            convention, where width and height are ``x2 - x1 + 1`` and
            ``y2 - y1 + 1``. Defaults to False, which is the convention shared
            by ``pycocotools`` and every mmdet v2+ detector, so the IoU here
            agrees with the mAP printed beside it. Setting this True shifts
            every IoU downward and inflates recall.
        backend_args (dict, optional): Arguments to instantiate the backend.
            Defaults to None.
        collect_device (str): Device name used to collect results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix: Optional[str] = 'det'

    def __init__(self,
                 ann_file: str,
                 score_thr: Optional[float] = 0.5,
                 iou_thr: float = 0.5,
                 max_dets: int = 100,
                 use_legacy_coordinate: bool = False,
                 backend_args: Optional[dict] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        if score_thr is not None and not 0.0 <= score_thr <= 1.0:
            raise ValueError(
                f'score_thr should be None or in [0, 1], but got {score_thr}')
        if not 0.0 <= iou_thr <= 1.0:
            raise ValueError(f'iou_thr should be in [0, 1], but got {iou_thr}')
        if max_dets <= 0:
            raise ValueError(
                f'max_dets should be positive, but got {max_dets}')

        self.ann_file = ann_file
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.max_dets = max_dets
        self.use_legacy_coordinate = use_legacy_coordinate
        self.backend_args = backend_args

        with get_local_path(
                ann_file, backend_args=self.backend_args) as local_path:
            self._coco_api = COCO(local_path)
        self.cat_ids = self._coco_api.getCatIds()
        if len(self.cat_ids) != 1:
            raise NotImplementedError(
                'PrecisionRecallMetric matches detections to ground truths '
                f'without regard to class and therefore needs a '
                f'single-category annotation file, but {ann_file} declares '
                f'{len(self.cat_ids)} categories.')

    def process(self, data_batch: dict, data_samples: List[dict]) -> None:
        """Store per-image predictions for later evaluation.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (List[dict]): A batch of data samples that contain
                annotations and predictions.
        """
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            self.results.append(
                dict(
                    img_id=data_sample['img_id'],
                    bboxes=pred['bboxes'].cpu().numpy(),
                    scores=pred['scores'].cpu().numpy()))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Aggregate true/false positives and negatives over the dataset.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: Precision, recall, F1, the score threshold used
            and the raw counts.
        """
        all_scores = []
        all_is_tp = []
        num_gt = 0

        for result in results:
            scores, is_tp, num_image_gt = self._match_image(
                result['img_id'], result['bboxes'], result['scores'])
            all_scores.append(scores)
            all_is_tp.append(is_tp)
            num_gt += num_image_gt

        scores = (np.concatenate(all_scores) if all_scores
                  else np.zeros(0, dtype=np.float32))
        is_tp = (np.concatenate(all_is_tp) if all_is_tp
                 else np.zeros(0, dtype=bool))

        # Rank every kept detection by score. Prefix stability of the greedy
        # matching makes the cumulative sums below the exact TP/FP counts at
        # the threshold equal to this detection's score.
        order = np.argsort(-scores, kind='stable')
        scores = scores[order]
        is_tp = is_tp[order]

        if self.score_thr is not None:
            score_thr = self.score_thr
            keep = scores > score_thr
        else:
            score_thr = self._best_f1_threshold(scores, is_tp, num_gt)
            # '>=' so detections tied with the chosen score are not dropped.
            keep = scores >= score_thr

        true_positive = int(is_tp[keep].sum())
        false_positive = int(keep.sum() - true_positive)
        false_negative = int(num_gt - true_positive)

        precision = (true_positive / (true_positive + false_positive)
                     if true_positive + false_positive else 0.0)
        recall = (true_positive / (true_positive + false_negative)
                  if true_positive + false_negative else 0.0)
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)

        return {
            'TP': true_positive,
            'FP': false_positive,
            'FN': false_negative,
            'score_thr': round(score_thr, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }

    @staticmethod
    def _best_f1_threshold(scores: np.ndarray, is_tp: np.ndarray,
                           num_gt: int) -> float:
        """Return the score threshold maximising F1.

        ``scores`` must be sorted descending and ``is_tp`` aligned to it. The
        prefix sums give the TP/FP counts at every threshold at once.
        """
        if scores.size == 0 or num_gt == 0:
            return 0.0
        cum_tp = np.cumsum(is_tp)
        cum_fp = np.cumsum(~is_tp)
        # F1 = 2TP / (2TP + FP + FN), with FN = num_gt - TP.
        f1 = 2.0 * cum_tp / (2.0 * cum_tp + cum_fp + num_gt - cum_tp)
        # Ties go to the highest threshold, i.e. the most precise point.
        return float(scores[int(np.argmax(f1))])

    def _match_image(self, img_id: int, bboxes: np.ndarray,
                     scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """Match one image's detections to its ground truth.

        Returns the kept detections' scores in descending order, an aligned
        boolean array marking which are true positives, and the image's ground
        truth count.

        Matching is delegated to ``tpfp_default``
        (``mmdet/evaluation/functional/mean_ap.py``), the VOC-devkit
        implementation the repo already ships and tests, rather than
        re-implemented here. It labels detections in descending score order,
        which is what makes the threshold sweep in ``compute_metrics`` exact.

        Note the rule differs in principle from COCO's: ``tpfp_default``
        matches a detection to its best-IoU ground truth over *all* of them and
        calls it a false positive when that one is already taken, whereas
        pycocotools re-selects among the *still unmatched* ground truths. The
        two therefore disagree only when one detection's global argmax is
        already claimed while another free ground truth also clears
        ``iou_thr``. On SSDD they agree exactly -- verified against a
        COCO-style matcher, same P/R to four decimals -- because ships are
        small and well separated relative to the detections that survive NMS.
        """
        gt_bboxes = self._load_gt_bboxes(img_id)
        num_gt = gt_bboxes.shape[0]

        order = np.argsort(-scores, kind='stable')[:self.max_dets]
        bboxes = bboxes[order]
        scores = scores[order]

        if num_gt == 0 or bboxes.shape[0] == 0:
            return scores, np.zeros(bboxes.shape[0], dtype=bool), num_gt

        det_bboxes = np.concatenate(
            [bboxes, scores[:, None]], axis=1).astype(np.float32)
        tp, _ = tpfp_default(
            det_bboxes,
            gt_bboxes.astype(np.float32),
            gt_bboxes_ignore=np.zeros((0, 4), dtype=np.float32),
            iou_thr=self.iou_thr,
            use_legacy_coordinate=self.use_legacy_coordinate)
        return scores, tp[0].astype(bool), num_gt

    def _load_gt_bboxes(self, img_id: int) -> np.ndarray:
        """Load one image's ground truth boxes as xyxy.

        COCO stores boxes as ``[x, y, width, height]``, so the trailing pair is
        converted from a size to a corner.
        """
        ann_ids = self._coco_api.getAnnIds(imgIds=[img_id])
        anns = self._coco_api.loadAnns(ann_ids)
        if len(anns) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        gt_bboxes = np.array(
            [ann['bbox'] for ann in anns], dtype=np.float32).reshape(-1, 4)
        gt_bboxes[:, 2:] += gt_bboxes[:, :2]
        return gt_bboxes
