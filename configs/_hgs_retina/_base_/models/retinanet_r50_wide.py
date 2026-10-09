_base_ = ['./retinanet_r50.py']

# RetinaNet-R50 at upstream widths.
#
# retinanet_r50.py only swaps the backbone into the R18 model definition, so it
# silently keeps an R18-sized FPN and head (64 channels). Measured cost of that
# inheritance:
#
#   ours  (64-wide)  : neck 1.56M + head 0.32M =  1.88M
#   upstream (256)   : neck 8.00M + head 4.82M = 12.82M
#
# i.e. 6.8x less detection capacity overall and 15x less in the bbox head --
# the part that actually classifies and regresses at every FPN level. Anchors,
# strides and FPN level count were already identical to upstream; only the width
# had been narrowed.
#
# Restoring 256 makes this a real RetinaNet-R50, so published numbers become
# comparable and the only remaining deviations are the data pipeline and
# test_cfg.
#
# Kept as a separate file rather than edited into retinanet_r50.py because that
# config is shared by every already-run R50 experiment; changing it would
# silently invalidate those results.
model = dict(
    neck=dict(out_channels=256),
    bbox_head=dict(in_channels=256, feat_channels=256))
