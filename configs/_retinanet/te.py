import torch

# 你的 ckpt 路径
ckpt_path = "/home1/zhangyn/code/mmdetection-3.0.0/tools/ckp/air/retina/epoch_175_student.pth"

ckpt = torch.load(ckpt_path, map_location="cpu")

print(ckpt['meta'])
