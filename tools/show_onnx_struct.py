# # import torch
# # from mmdet.apis import init_detector
# # from mqbench.utils.state import enable_quantization

# # # 1. 初始化（会自动触发你改好的 checkpoint.py 注入逻辑）
# # config = 'configs/_retinanet/ssdd_retina_r18.py'
# # checkpoint = 'tools/quanti/retinanet/SSDD/testpth_02_10_07_40/epoch_1.pth'
# # model = init_detector(config, checkpoint, device='cpu')

# # # 2. 强制开启量化状态
# # model.eval()
# # enable_quantization(model)

# # # 3. 原生导出
# # dummy_input = torch.randn(1, 3, 800, 800)
# # torch.onnx.export(
# #     model, dummy_input, "raw_test_qdq.onnx",
# #     opset_version=13, # 必须 13
# #     do_constant_folding=True,
# #     input_names=['input'], output_names=['output']
# # )
# # print("✅ 导出完成，请再次运行查看脚本打印这个 raw_test_qdq.onnx 的结构。")


# import onnx
# from collections import Counter

# def check_onnx_structure(model_path):
#     # 1. 加载模型
#     model = onnx.load(model_path)
#     graph = model.graph
    
#     print(f"============================================================")
#     print(f"📊 ONNX 模型概览: {model_path}")
#     print(f"============================================================")
    
#     # 2. 统计所有算子类型
#     op_types = [node.op_type for node in graph.node]
#     op_counts = Counter(op_types)
    
#     print("📈 算子分布统计:")
#     for op, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
#         print(f"  - {op}: {count}")
    
#     print(f"\n核心量化节点检查:")
#     q_count = op_counts.get('QuantizeLinear', 0)
#     dq_count = op_counts.get('DequantizeLinear', 0)
#     print(f"  - [QuantizeLinear]: {q_count}")
#     print(f"  - [DequantizeLinear]: {dq_count}")
    
#     # 3. 打印前 20 个节点的详细流转 (方便观察输入附近的量化)
#     print(f"\n🔬 前 20 个节点流转明细 (观察是否有 Q/DQ 夹在 Conv 中):")
#     print(f"{'Op Type':<20} | {'Input':<30} | {'Output':<30}")
#     print("-" * 85)
    
#     for i, node in enumerate(graph.node[:20]):
#         inputs = ", ".join(node.input[:2])  # 只打印前两个输入
#         outputs = ", ".join(node.output)
#         print(f"{node.op_type:<20} | {inputs:<30} | {outputs:<30}")

#     if q_count == 0 and 'Mul' in op_counts:
#         print(f"\n⚠️ 警告: 发现大量 Mul 节点但没有 Q 节点。")
#         print(f"这通常意味着量化逻辑被退化成了普通的浮点乘法。")
#         print(f"请检查导出时的 opset_version 是否 >= 13。")

# if __name__ == "__main__":
#     # 替换成你生成的 ONNX 文件路径
#     check_onnx_structure('/home1/zhangyn/code/mmdetection-3.0.0/mmdeploy/tools/torch2onnx/end2end.onnx')

import onnxruntime as ort
import numpy as np
import torch

# 1. 加载 ONNX
onnx_path = "/home1/zhangyn/code/mmdetection-3.0.0/mmdeploy/tools/torch2onnx/end2end.onnx"
session = ort.InferenceSession(onnx_path)

# 2. 构造一个假输入 (注意：一定要用 4D，虽然我们修了模型，但标准输入还是 4D)
# 假设你的输入是 800x800
input_shape = (1, 3, 800, 800)
dummy_input = np.random.randn(*input_shape).astype(np.float32)

# 3. 推理
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: dummy_input})

# 4. 检查输出
print(">>> 输出层数量:", len(outputs))
for i, out in enumerate(outputs):
    print(f"Output {i} shape: {out.shape}")
    print(f"Output {i} range: [{out.min()}, {out.max()}]")
    
    # 如果输出里有 NaN，说明精度崩了；如果有正常的数值，说明大概率成功了。
    if np.isnan(out).any():
        print("❌ 警告：检测到 NaN！模型计算可能有问题。")
    else:
        print("✅ 数值正常。")