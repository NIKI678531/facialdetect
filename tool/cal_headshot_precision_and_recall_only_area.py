import os
import re

# 定义文件夹路径
dir1 = "/Users/dingpengxu1/Documents/duet_大头照数据_2025_01_14_宽比_高比_面积比_分类别"
subfolders = ["大头照", "非大头照"]

# 初始化结果存储
results = []

# 遍历 area_threshold 从 0.01 到 1.0
for area_threshold in [i * 0.01 for i in range(1, 101)]:
    # 初始化统计变量
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # 遍历子文件夹
    for subfolder in subfolders:
        folder_path = os.path.join(dir1, subfolder)
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpeg"):
                # 解析文件名中的面积比例
                parts = filename.split('_')
                if len(parts) >= 6:  # 格式为 img_283199_1_w0.53_h0.47_a0.25.jpeg
                    a_str = parts[5]
                    a_match = re.search(r'a(\d+\.\d+)', a_str)
                    if a_match:
                        a = float(a_match.group(1))
                    else:
                        a = 1.0  # 如果匹配不到，设为1.0
                else:  # 格式为 img_283208_5.jpeg，面积比例为 1
                    a = 1.0

                # 判断是否为大头照（仅根据面积比例）
                is_big_head = a >= area_threshold

                # 根据子文件夹名称和判断结果更新统计变量
                if subfolder == "大头照":
                    if is_big_head:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    if is_big_head:
                        false_positives += 1

    # 计算精准度和召回率
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    # 存储结果
    results.append((area_threshold, precision, recall))

# 打印结果
print("area_threshold\tPrecision\tRecall")
for area_threshold, precision, recall in results:
    print(f"{area_threshold:.2f}\t\t{precision:.2f}\t\t{recall:.2f}")