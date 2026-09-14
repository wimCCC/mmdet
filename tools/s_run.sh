RetinaNet
python tools/train_quanti.py configs/_retinanet/retinanet_r18_fpn_nwpu_quanti.py


FCOS
python tools/train_quanti.py configs/_fcos/fcos_center-normbbox-centeronreg-giou_r18_fpn_gn-head_1x_nwpu_quanti.py
ppq_test
python tools/train_quanti_ppq.py configs/_fcos/fcos_center-normbbox-centeronreg-giou_r18_fpn_gn-head_1x_nwpu_ppq.py