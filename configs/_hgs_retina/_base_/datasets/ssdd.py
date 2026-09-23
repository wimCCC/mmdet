# SSDD COCO-format dataset definition shared by all HGS RetinaNet experiments.
data_root = (
    '/data5/caiwm/mmdet/data/SSDD/raw/Official-SSDD-OPEN/'
    'BBox_SSDD/coco_style/')

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

val_evaluator = dict(
    type='CocoMetric', ann_file=data_root + 'annotations/test.json')
test_evaluator = val_evaluator
