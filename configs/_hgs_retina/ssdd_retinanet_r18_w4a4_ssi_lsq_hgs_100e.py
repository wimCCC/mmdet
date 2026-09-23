_base_ = [
    './_base_/datasets/ssdd.py',
    './_base_/models/retinanet_r18.py',
    './_base_/quantization/ssi_lsq_hgs_w4a4.py',
    './_base_/schedules/ssdd_100e.py',
    './_base_/runtime.py',
]

work_dir = 'work_dirs/hgs_retina/ssdd_retinanet_r18_w4a4_ssi_lsq_hgs_100e'
