# SSDD COCO-format dataset definition shared by all HGS RetinaNet experiments.
import os.path as osp

# The same tree has lived under /data5 on the original training machine and
# under this repository's gitignored data/ elsewhere, so pick whichever exists.
_data_root_candidates = [
    '/data5/caiwm/mmdet/data/SSDD/raw/Official-SSDD-OPEN/'
    'BBox_SSDD/coco_style/',
    '/home2/caiwm/mmdet/data/SSDD/raw/Official-SSDD-OPEN/'
    'BBox_SSDD/coco_style/',
]
data_root = next(
    (path for path in _data_root_candidates if osp.isdir(path)),
    _data_root_candidates[0])

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        pipeline=train_pipeline,
        metainfo=dict(classes=('ship',), palette=[(220, 20, 60)])))
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/test/'),
        pipeline=test_pipeline,
        metainfo=dict(classes=('ship',), palette=[(220, 20, 60)])))
test_dataloader = val_dataloader

# CocoMetric's ``ship_precision`` field is not a precision -- it is COCO's
# precision array averaged over every IoU threshold and recall level, i.e. AP
# printed under a second name (it comes out identical to coco/bbox_mAP). P/R
# have to be measured at an operating point, hence the second metric: matched at
# IoU 0.5, capped at maxDets to mirror the detector's test_cfg.max_per_img.
#
# score_thr=None selects the threshold maximising F1 instead of pinning 0.5.
# At 0.5 these detectors only misfire 15-32 times while missing 62-79 of the 546
# ships, so the operating point sits on the high-precision side of the curve and
# understates recall.
#
# Note the F1-optimal point is NOT the recall ceiling: F1 turns over before
# recall saturates. At IoU 0.5 recall keeps rising to ~0.96 as the threshold
# goes to 0, at the cost of precision collapsing to ~0.02. P and R therefore
# trade off and cannot both be pushed to 0.93-0.94 by any choice of threshold --
# they cross near 0.91. The chosen threshold is reported as ``det/score_thr``;
# read it, because it differs per run and P/R from different thresholds are not
# one common operating point.
#
# These knobs CANNOT be changed from the command line. --cfg-options merges into
# the parsed config, so ``val_evaluator.1.iou_thr=0.4`` is silently ignored, and
# rebinding a helper name below does not help either -- the dict literal was
# already evaluated when the config was parsed (verified: it changes
# ``pr_iou_thr`` in the dumped config while ``iou_thr`` stays at its old value).
# To vary them, edit this file or make a wrapper config that replaces
# ``test_evaluator`` outright.
pr_score_thr = None
pr_iou_thr = 0.5
pr_max_dets = 100

val_evaluator = [
    dict(type='CocoMetric', ann_file=data_root + 'annotations/test.json'),
    dict(
        type='PrecisionRecallMetric',
        ann_file=data_root + 'annotations/test.json',
        score_thr=pr_score_thr,
        iou_thr=pr_iou_thr,
        max_dets=pr_max_dets),
]
test_evaluator = val_evaluator
