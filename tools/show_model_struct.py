import torch
import os

# ================= 配置 =================
# 你的权重路径
pth_path = '/home1/zhangyn/code/mmdetection-3.0.0/tools/quanti/retinanet/SSDD/testpth_02_10_07_40/epoch_1.pth'
# 输出的文本文件路径 (方便你查看)
log_file = 'model_structure_log.txt'
# =======================================

def inspect_checkpoint():
    print(f"🚀 正在加载: {pth_path}")
    
    if not os.path.exists(pth_path):
        print("❌ 文件不存在！")
        return

    # 1. 加载权重
    try:
        checkpoint = torch.load(pth_path, map_location='cpu')
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # 2. 找到 state_dict (权重字典)
    state_dict = None
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            print("✅ 发现 'state_dict' 键，正在读取...")
            state_dict = checkpoint['state_dict']
        else:
            print("⚠️ 没找到 'state_dict'，假设整个字典就是权重...")
            state_dict = checkpoint
    else:
        print("❌ Checkpoint 格式不对，不是字典。")
        return

    # 3. 统计和筛选
    all_keys = list(state_dict.keys())
    qat_keys = []
    normal_keys = []

    # 关键词：MQBench 或 PyTorch QAT 通常包含这些
    qat_keywords = ['scale', 'zero_point', 'observer', 'fake_quant', '_amax', 'quant']

    print(f"\n📊 统计信息:")
    print(f"   总参数量 (Keys): {len(all_keys)}")

    for key in all_keys:
        if any(k in key for k in qat_keywords):
            qat_keys.append(key)
        else:
            normal_keys.append(key)

    # 4. 屏幕输出结果
    print(f"   --------------------------------")
    print(f"   🔹 普通参数 (Weight/Bias): {len(normal_keys)} 个")
    print(f"   🔸 量化参数 (Scale/ZP/Obs): {len(qat_keys)} 个")
    print(f"   --------------------------------")

    if len(qat_keys) > 0:
        print("\n🎉 【好消息】发现了量化参数！你的 QAT 是成功的！")
        print("   以下是前 10 个量化参数示例：")
        for i, k in enumerate(qat_keys[:10]):
            print(f"   {i+1}. {k}  (Shape: {state_dict[k].shape})")
    else:
        print("\n😱 【坏消息】一个量化参数都没找到...")
        print("   这说明 .pth 里只存了 FP32 权重 (Weight)，没存 QAT 参数。")
        print("   结论：你的训练是生效的(因为精度有变化)，但保存时丢失了 Observer 信息。")

    # 5. 保存完整列表到文件
    with open(log_file, 'w') as f:
        f.write(f"=== Checkpoint Analysis for {os.path.basename(pth_path)} ===\n")
        f.write(f"Total Keys: {len(all_keys)}\n")
        f.write(f"QAT Keys Found: {len(qat_keys)}\n\n")
        
        f.write("--- [QAT Parameters] ---\n")
        for k in qat_keys:
            f.write(f"{k}\n")
            
        f.write("\n--- [Normal Parameters] ---\n")
        for k in normal_keys:
            f.write(f"{k}\n")
            
    print(f"\n💾 完整 Key 列表已保存到: {log_file}")
    print("   你可以打开这个文件搜一下看看具体有什么。")

if __name__ == '__main__':
    inspect_checkpoint()