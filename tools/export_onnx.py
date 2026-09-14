import argparse
import torch
import torch.nn as nn
import onnx
import os
from mmengine.config import Config
from mmdet.apis import init_detector

# ✅ MQBench 依赖
try:
    from mqbench.prepare_by_platform import prepare_by_platform, BackendType
    from mqbench.fake_quantize.quantize_base import QuantizeBase
except ImportError:
    raise ImportError("请确保安装了 MQBench: pip install mqbench")

def parse_args():
    parser = argparse.ArgumentParser(description='Export QAT ONNX (Academic White-box Mode)')
    default_config = '/home1/zhangyn/code/mmdetection-3.0.0/configs/_retinanet/ssdd_retina_r18.py'
    default_checkpoint = '/home1/zhangyn/code/mmdetection-3.0.0/tools/quanti/retinanet/SSDD/testpth_02_10_07_40/epoch_1.pth'
    default_output = '/home1/zhangyn/code/mmdetection-3.0.0/tools/ckp/ssdd/onnx/retinanet_qdq.onnx'

    parser.add_argument('config', nargs='?', default=default_config)
    parser.add_argument('checkpoint', nargs='?', default=default_checkpoint)
    parser.add_argument('output', nargs='?', default=default_output)
    parser.add_argument('--input-shape', type=int, nargs='+', default=[800, 800])
    args = parser.parse_args()
    return args

# ==============================================================================
# 1. 强制绘图算子 (Symbolic Force)
# ==============================================================================
class ForceQDQFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, scale, zero_point):
        # 强制命令 ONNX 画出 QuantizeLinear 节点
        return g.op("QuantizeLinear", x, scale, zero_point)
        
    @staticmethod
    def forward(ctx, x, scale, zero_point):
        # 这里的计算不重要，因为会被后面的 DQ 抵消，主要是为了过 ONNX 检查
        return x / scale + zero_point

class ForceDeQDQFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, x, scale, zero_point):
        # 强制命令 ONNX 画出 DequantizeLinear 节点
        return g.op("DequantizeLinear", x, scale, zero_point)

    @staticmethod
    def forward(ctx, x, scale, zero_point):
        return (x - zero_point) * scale

# ==============================================================================
# 2. 白盒替换模块 (Compatible Module)
# ==============================================================================
class WhiteBoxQDQ(nn.Module):
    def __init__(self, scale, zero_point):
        super().__init__()
        # 确保参数格式正确 (1D Tensor)
        s = scale.detach().clone().view(1)
        z = zero_point.detach().clone().view(1).to(torch.uint8) # ONNX要求uint8
        
        self.register_buffer('scale', s)
        self.register_buffer('zero_point', z)

    def forward(self, x):
        # 显式调用强制算子
        q = ForceQDQFunction.apply(x, self.scale, self.zero_point)
        dq = ForceDeQDQFunction.apply(q, self.scale, self.zero_point)
        return dq

    @staticmethod
    def from_mqbench(mq_node):
        if not hasattr(mq_node, 'scale') or mq_node.scale is None: return None
        s = mq_node.scale
        z = mq_node.zero_point if hasattr(mq_node, 'zero_point') else torch.zeros_like(s)
        return WhiteBoxQDQ(s, z)

# ==============================================================================
# 3. 递归替换逻辑
# ==============================================================================
def replace_nodes_recursive(model):
    count = 0
    for name, child in model.named_children():
        # 检查是否是 MQBench 节点 (只要类名沾边就算)
        if isinstance(child, QuantizeBase) or 'FakeQuant' in child.__class__.__name__:
            new_node = WhiteBoxQDQ.from_mqbench(child)
            if new_node is not None:
                setattr(model, name, new_node)
                count += 1
        else:
            count += replace_nodes_recursive(child)
    return count

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1. 加载配置 & FP32模型
    print("🚀 加载配置...")
    cfg = Config.fromfile(args.config)
    if 'model' in cfg and 'backbone' in cfg.model:
        if 'norm_eval' in cfg.model.backbone: cfg.model.backbone.norm_eval = False
        if 'frozen_stages' in cfg.model.backbone: cfg.model.backbone.frozen_stages = -1
    cfg.model.init_cfg = None 
    model = init_detector(cfg, checkpoint=None, device='cpu')
    if hasattr(model, 'forward_dummy'): model.forward = model.forward_dummy

    # 2. MQBench Prepare (使用 Academic 模式！)
    # ⚠️ 关键：Academic 模式不会生成 GraphModule黑盒，而是普通的 nn.Module
    print("🔧 重建结构 (White-box Mode)...")
    model.train() 
    
    # 补全所有可能缺失的配置，防止 KeyError
    extra_qconfig_dict = {
        'w_observer': 'MinMaxObserver',
        'a_observer': 'EMAV2Observer',
        'w_fakequantize': 'LearnableFakeQuantize',
        'a_fakequantize': 'LearnableFakeQuantize',
        # Academic 模式需要的详细配置
        'w_qscheme': {
            'bit': 8, 'symmetry': True, 'per_channel': False, 'pot_scale': False
        },
        'a_qscheme': {
            'bit': 8, 'symmetry': True, 'per_channel': False, 'pot_scale': False
        }
    }
    
    try:
        # 使用 Academic 后端
        prepare_by_platform(model, BackendType.Academic, extra_qconfig_dict)
        print("✅ MQBench Structure Prepared (Academic).")
    except Exception as e:
        print(f"❌ Prepare 失败: {e}")
        return

    # 3. 加载权重
    print(f"📥 加载权重...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    clean_state_dict = {}
    if hasattr(model, 'state_dict'): model_keys = set(model.state_dict().keys())
    else: model_keys = set(model.module.state_dict().keys())
    
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        if name in model_keys: clean_state_dict[name] = v
        
    model.load_state_dict(clean_state_dict, strict=False)
    print("✅ 权重加载完毕。")

    # 🛑 4. 执行白盒替换
    print("🏥 [手术] 开始替换节点...")
    # 由于是 Academic 模式，模型是标准的 Tree 结构，递归替换 100% 有效
    count = replace_nodes_recursive(model)
    print(f"✅ 替换完成！共替换 {count} 个节点。")
    
    if count == 0:
        print("❌ 警告：替换数为 0。请检查 model 结构 (print(model))。")

    # 5. 导出 ONNX
    print("\n🔌 导出 ONNX (Force Q/DQ)...")
    model.eval()
    input_shape = (1, 3, args.input_shape[0], args.input_shape[1])
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model,       
        dummy_input,
        args.output,
        input_names=['input'], output_names=['output'],
        opset_version=13,
        do_constant_folding=True, 
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    
    if os.path.exists(args.output):
        import onnx
        onnx_model = onnx.load(args.output)
        ops = [n.op_type for n in onnx_model.graph.node]
        print("="*60)
        print(f"🎉 文件已生成: {args.output}")
        print(f"Q 节点数: {ops.count('QuantizeLinear')}")
        print(f"DQ 节点数: {ops.count('DequantizeLinear')}")
        print("="*60)

if __name__ == '__main__':
    main()