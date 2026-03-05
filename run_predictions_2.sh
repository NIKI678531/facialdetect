#!/bin/bash

LOG_FILE="predict_log_2.log"

# 清空旧日志（可选）
> "$LOG_FILE"

for i in {2..2}
do
    echo "正在处理序号: $i" | tee -a "$LOG_FILE"  # tee 会同时输出到终端和日志
    python predict_single_pic.py CockPicture all model_weight/isCockPicture/CockPicture_image_prediction_model_9838_9612_88_best_f1_model.pth "/Volumes/移动硬盘1TB/duet/duet_user_picture_2025_01_13_new/duet_user_picture_2025_01_13_${i}_new" 0.88 1 all | tee -a "$LOG_FILE"
done

echo "全部完成！日志已保存到: $LOG_FILE"