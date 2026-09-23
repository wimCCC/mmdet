_base_ = ['./retinanet_r18.py']

# RetinaNet-50 variant: only backbone depth and FPN input channels differ.
model = dict(
    backbone=dict(
        depth=50,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(in_channels=[256, 512, 1024, 2048]))
