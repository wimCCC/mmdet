#!/bin/bash

# 指定显存空闲容量阈值，单位为MiB
FREE_MEMORY_THRESHOLD=22000  # 根据需要修改这个值

# 指定要监控的GPU ID
GPUS_TO_MONITOR=(0)

# 指定要运行的Python程序及其参数的数组
PYTHON_COMMANDS=(
    # "python tools/train_quanti.py configs/_retinanet/retinanet_r18_fpn_visdrone_quanti_0.py"
    # "python tools/train_quanti.py configs/_retinanet/retinanet_r18_fpn_visdrone_quanti_1.py"
    "python tools/train_quanti.py configs/_fcos/fcos_dior_quanti_1.py"
    # "python tools/train_quanti.py configs/_fasterRcnn/frcnn_vis_quanti_0.py"
    # "python tools/train_quanti.py configs/_fasterRcnn/frcnn_vis_quanti_2.py"
    # 在这里添加更多的Python命令，每个命令用引号括起来，每个命令为数组的一个元素
)

# 连续检测次数阈值，等于4表示2分钟（每30秒检查一次）
CONSECUTIVE_CHECKS_REQUIRED=1

# 无限循环，每30秒检查一次
while true; do
  for GPU_ID in "${GPUS_TO_MONITOR[@]}"; do
    consecutive_checks=0  # 初始化连续检测计数器
    while [[ "$consecutive_checks" -lt "$CONSECUTIVE_CHECKS_REQUIRED" ]]; do
      # 使用nvidia-smi命令获取指定GPU ID的空闲显存容量
      FREE_MEMORY=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU_ID | cut -f1 -d' ')

      # 检查空闲显存是否超过阈值
      if [[ "$FREE_MEMORY" -gt "$FREE_MEMORY_THRESHOLD" ]]; then
        ((consecutive_checks++))
        echo "GPU $GPU_ID: 检测到足够的空闲显存：${FREE_MEMORY}MiB，连续${consecutive_checks}次检测通过。"
        if [[ "$consecutive_checks" -eq "$CONSECUTIVE_CHECKS_REQUIRED" ]]; then
          echo "GPU $GPU_ID: 连续${CONSECUTIVE_CHECKS_REQUIRED}次检测到足够的空闲显存，开始依次执行Python命令..."
          # 遍历PYTHON_COMMANDS数组，依次执行每个Python命令
          for CMD in "${PYTHON_COMMANDS[@]}"; do
            echo "正在执行: $CMD"
            CUDA_VISIBLE_DEVICES=$GPU_ID eval $CMD
          done
          # 如果不想在所有程序运行结束后继续监控，可以取消下面一行的注释
          break 3
        fi
      else
        consecutive_checks=0  # 如果检测失败，重置计数器
        echo "GPU $GPU_ID: 显存不足：当前空闲${FREE_MEMORY}MiB，需要${FREE_MEMORY_THRESHOLD}MiB。等待下一次检查..."
        break  # 跳出内层循环，等待下一次外层循环的检查
      fi

      # 等待30秒
      sleep 30
    done
  done
done
