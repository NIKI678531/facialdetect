import os
import re

def extract_brightness(filename):
    """从文件名中提取亮度值"""
    match = re.search(r'brightness_(\d+\.\d+)', filename)
    if match:
        return float(match.group(1))
    return None

def calculate_precision_recall(dark_folder, normal_folder, thresholds):
    """计算不同阈值下的精准度和召回率"""
    dark_images = [f for f in os.listdir(dark_folder) if f.endswith('.jpeg')]
    normal_images = [f for f in os.listdir(normal_folder) if f.endswith('.jpeg')]

    results = []

    for threshold in thresholds:
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # 计算昏暗图片的TP和FN
        for image in dark_images:
            brightness = extract_brightness(image)
            if brightness is not None:
                if brightness < threshold:
                    true_positives += 1
                else:
                    false_negatives += 1

        # 计算正常图片的FP
        for image in normal_images:
            brightness = extract_brightness(image)
            if brightness is not None:
                if brightness < threshold:
                    false_positives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        # 计算F1分数
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results.append((threshold, precision, recall, f1_score))

    return results


# 文件夹路径
dark_folder = '/Users/dingpengxu1/Documents/duet_测试图片_图片昏暗_2025_01_15_带亮度/昏暗图片'
normal_folder = '/Users/dingpengxu1/Documents/duet_测试图片_图片昏暗_2025_01_15_带亮度/正常图片'

# 定义阈值范围
thresholds = [i * 0.01 for i in range(101)]  # 0.0, 0.1, ..., 1.0

# 计算精准度、召回率和F1分数
results = calculate_precision_recall(dark_folder, normal_folder, thresholds)

# 找到最优的F1分数及其对应的精准度、召回率和阈值
best_f1_score = 0
best_result = None

for result in results:
    threshold, precision, recall, f1_score = result
    if f1_score > best_f1_score:
        best_f1_score = f1_score
        best_result = result

# 打印最优F1分数时的精准度、召回率和阈值
if best_result:
    threshold, precision, recall, f1_score = best_result
    print(f"Best F1 Score: {f1_score:.2f}")
    print(f"Threshold: {threshold:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
else:
    print("No valid results found.")