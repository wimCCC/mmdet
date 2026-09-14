python tools/train_quanti.py configs/_fasterRcnn/faster_rcnn_r18_fpn_nwpu_quanti.py
python tools/train_quanti.py configs/_fcos/fcos_center-normbbox-centeronreg-giou_r18_fpn_gn-head_1x_nwpu_quanti.py
python tools/train_quanti.py configs/_retinanet/retinanet_r18_fpn_nwpu_quanti.py


CUDA_VISIBLE_DEVICES=1 python tools/train_quanti.py configs/_fcos/fcos_vis_quanti_0.py

dior
python tools/train_quanti.py configs/_retinanet/retinanet_r18_fpn_dior_quanti.py
