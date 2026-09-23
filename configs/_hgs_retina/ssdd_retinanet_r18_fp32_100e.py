_base_ = [
    './_base_/datasets/ssdd.py',
    './_base_/models/retinanet_r18.py',
    './_base_/schedules/ssdd_100e.py',
    './_base_/runtime.py',
]

# Full-precision control with the same data, schedule, and seed.
runner_type = 'Runner'
model = dict(type='RetinaNet')
extra_config = None
qat_compensation = dict(enabled=False)
custom_hooks = []

work_dir = 'work_dirs/hgs_retina/ssdd_retinanet_r18_fp32_100e'
