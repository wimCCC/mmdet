import onnxruntime as ort
import numpy as np
import cv2
import os
import glob

# ================= 核心配置区域 =================
ONNX_PATH = "/home1/zhangyn/code/mmdetection-3.0.0/mmdeploy/tools/torch2onnx/end2end_cache.onnx"
IMG_DIR = "/home1/zhangyn/code/mmdetection-3.0.0/data/SSDD/images/test"
OUTPUT_DIR = "/home1/zhangyn/code/mmdetection-3.0.0/mmdeploy/tools/torch2onnx/test_results"

INPUT_W, INPUT_H = 1344, 800
SCORE_THR = 0.5
NMS_THR = 0.2

# --- 终极宽高调整参数 ---
# 如果你的 mmdet 配置文件里 bbox_coder 的 target_stds 被改过，请在这里对应修改
# (例如改成了 [0.1, 0.1, 0.2, 0.2])，默认是 [1.0, 1.0, 1.0, 1.0]
TARGET_STDS = [1.0, 1.0, 1.0, 1.0]

# 针对 INT8 量化导致的回归头指数膨胀误差，强制收紧宽度
# 1.0 表示不缩放，0.85 表示宽度强制缩小 15%
INT8_WIDTH_CALIB = 0.85  
# ===============================================

def sigmoid(x):
    # 增加数值稳定性的 sigmoid，解决 overflow 警告
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

def decode_bbox(anchors, deltas, mean=[0,0,0,0], std=TARGET_STDS):
    """将回归偏移量应用到锚框上"""
    dx = deltas[:, 0] * std[0] + mean[0]
    dy = deltas[:, 1] * std[1] + mean[1]
    dw = deltas[:, 2] * std[2] + mean[2]
    dh = deltas[:, 3] * std[3] + mean[3]

    gw = anchors[:, 2] - anchors[:, 0]
    gh = anchors[:, 3] - anchors[:, 1]
    gx = anchors[:, 0] + gw * 0.5
    gy = anchors[:, 1] + gh * 0.5

    # 应用 INT8 量化宽度矫正系数
    pw = gw * np.exp(dw) * INT8_WIDTH_CALIB
    ph = gh * np.exp(dh)
    
    px = gx + dx * gw
    py = gy + dy * gh

    x1 = px - pw * 0.5
    y1 = py - ph * 0.5
    x2 = px + pw * 0.5
    y2 = py + ph * 0.5
    
    return np.stack([x1, y1, x2, y2], axis=-1)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"🚀 加载 ONNX 模型: {ONNX_PATH}")
    sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    image_paths = glob.glob(os.path.join(IMG_DIR, '*.jpg'))
    if not image_paths:
        print(f"⚠️ 警告: 在 {IMG_DIR} 未找到任何 .jpg 图片！")
        return
        
    print(f"📂 找到 {len(image_paths)} 张图片，开始批量推理...\n")

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            print(f"读取失败: {img_name}，跳过。")
            continue
        
        h_ori, w_ori = raw_img.shape[:2]

        # 1. 【预处理】等比例缩放与 Padding
        scale = min(INPUT_W / w_ori, INPUT_H / h_ori)
        new_w, new_h = int(w_ori * scale), int(h_ori * scale)
        resized_img = cv2.resize(raw_img, (new_w, new_h))
        
        # BGR 转换为 RGB
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # 构建背景为黑色的画板进行 Pad
        padded_img = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.float32)
        padded_img[:new_h, :new_w, :] = rgb_img
        
        # 归一化 (MMDetection ImageNet 均值和方差)
        padded_img = (padded_img - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
        img_tensor = padded_img.transpose(2, 0, 1)[None, :].astype(np.float32)

        # 2. 【推理】
        outputs = sess.run(None, {input_name: img_tensor})

        cls_maps, reg_maps = [], []
        for out in outputs:
            out = np.squeeze(out)
            if out.ndim == 3:
                if out.shape[0] == 9: cls_maps.append(out)
                elif out.shape[0] == 36: reg_maps.append(out)

        if len(cls_maps) == 0 or len(cls_maps) != len(reg_maps):
            print(f"[{img_name}] 警告：分类头和回归头不匹配或未找到，跳过。")
            continue

        proposals, scores_list = [], []
        strides = [8, 16, 32, 64, 128]
        
        # 3. 【解码与锚框生成】
        for c_map, r_map, stride in zip(cls_maps, reg_maps, strides):
            h, w = c_map.shape[1], c_map.shape[2]
            
            # 生成 Base Anchors (严格按照 MMDetection 源码 w/h ratio 定义)
            base_anchors = []
            scales = np.array([1, 2**(1/3), 2**(2/3)]) * 4
            ratios = np.array([0.5, 1.0, 2.0])
            for s in scales:
                for r in ratios:
                    base_w = s * stride * np.sqrt(r)
                    base_h = s * stride / np.sqrt(r)
                    base_anchors.append([-base_w/2, -base_h/2, base_w/2, base_h/2])
            base_anchors = np.array(base_anchors)

            # 左上角基准点的网格偏移
            sx = np.arange(w) * stride
            sy = np.arange(h) * stride
            sx, sy = np.meshgrid(sx, sy)
            
            grid_anchors = np.zeros((h, w, 9, 4))
            grid_anchors[..., 0] = sx[..., None] + base_anchors[:, 0]
            grid_anchors[..., 1] = sy[..., None] + base_anchors[:, 1]
            grid_anchors[..., 2] = sx[..., None] + base_anchors[:, 2]
            grid_anchors[..., 3] = sy[..., None] + base_anchors[:, 3]
            flat_anchors = grid_anchors.reshape(-1, 4)
            
            # 处理分类和回归
            c_map = sigmoid(c_map)
            flat_scores = c_map.transpose(1, 2, 0).reshape(-1)
            flat_deltas = r_map.transpose(1, 2, 0).reshape(h, w, 9, 4).reshape(-1, 4)
            
            # 分数过滤
            keep = flat_scores > SCORE_THR
            if not np.any(keep): continue
            
            # 边框解码
            decoded_boxes = decode_bbox(flat_anchors[keep], flat_deltas[keep])
            
            # 限制边界
            decoded_boxes[:, 0::2] = np.clip(decoded_boxes[:, 0::2], 0, INPUT_W - 1)
            decoded_boxes[:, 1::2] = np.clip(decoded_boxes[:, 1::2], 0, INPUT_H - 1)
            
            proposals.append(decoded_boxes)
            scores_list.append(flat_scores[keep])

        if not proposals:
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), raw_img)
            print(f"✅ [{img_name}] 未检测到目标。")
            continue

        proposals = np.concatenate(proposals, axis=0)
        scores = np.concatenate(scores_list, axis=0)

        # 4. 【NMS 去重】
        boxes_xywh = proposals.copy()
        boxes_xywh[:, 2] = proposals[:, 2] - proposals[:, 0]
        boxes_xywh[:, 3] = proposals[:, 3] - proposals[:, 1]
        
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh.tolist(),
            scores=scores.tolist(),
            score_threshold=SCORE_THR,
            nms_threshold=NMS_THR
        )
        
        # 5. 【坐标还原并画图】
        draw_img = raw_img.copy()
        for i in indices:
            idx = int(i)
            box = proposals[idx]
            score = scores[idx]
            
            x1 = int(box[0] / scale)
            y1 = int(box[1] / scale)
            x2 = int(box[2] / scale)
            y2 = int(box[3] / scale)
            
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(draw_img, f"{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        save_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(save_path, draw_img)
        print(f"✅ [{img_name}] 检出目标: {len(indices)} 个")

    print(f"\n🎉 批量测试完成！所有结果已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()