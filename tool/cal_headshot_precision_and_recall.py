import os
import csv
from tqdm import tqdm


# def parse_image_info(filename):
#     """从文件名中解析出宽度、高度和面积比例"""
#     parts = filename.split('_')
#     if len(parts) == 3:
#         # 处理格式为 img_283208_5.jpeg 的情况
#         width_ratio = 1.0
#         height_ratio = 1.0
#         area_ratio = 1.0
#     else:
#         # 处理格式为 img_283199_1_w0.53_h0.47_a0.25.jpeg 的情况
#         width_ratio = float(parts[3][1:])
#         height_ratio = float(parts[4][1:])
#         area_ratio = float(parts[5][1:].split('.')[0])
#         print(filename, width_ratio, height_ratio, area_ratio, parts[5][1:].split('.')[1])
#     return width_ratio, height_ratio, area_ratio

import re

def parse_image_info(filename):
    """从文件名中解析出宽度、高度和面积比例"""
    pattern = re.compile(r'_w(\d+\.\d+)_h(\d+\.\d+)_a(\d+\.\d+).jpeg$')
    match = pattern.search(filename)
    if match:
        width_ratio = float(match.group(1))
        height_ratio = float(match.group(2))
        area_ratio = float(match.group(3))
    else:
        # 如果没有匹配到，设为1.0
        width_ratio = 1.0
        height_ratio = 1.0
        area_ratio = 1.0
    print(filename, width_ratio, height_ratio, area_ratio)
    return width_ratio, height_ratio, area_ratio

def load_image_data(dir_path):
    """加载所有图片的数据"""
    image_data = []
    for category in ['大头照', '非大头照']:
        category_path = os.path.join(dir_path, category)
        for filename in os.listdir(category_path):
            if filename.endswith('.jpeg'):
                width_ratio, height_ratio, area_ratio = parse_image_info(filename)
                # print(filename, width_ratio, height_ratio, area_ratio)
                image_data.append({
                    'width_ratio': width_ratio,
                    'height_ratio': height_ratio,
                    'area_ratio': area_ratio,
                    'category': category
                })
    return image_data


def calculate_metrics(image_data, width_threshold, height_threshold, area_threshold):
    """计算精准度和召回率"""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    for image in image_data:
        # print(image, image['width_ratio'], image['height_ratio'], image['area_ratio'])
        is_headshot = ((image['width_ratio'] >= width_threshold or
                       image['height_ratio'] >= height_threshold) and
                       image['area_ratio'] >= area_threshold)
        if image['category'] == '大头照':
            if is_headshot:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if is_headshot:
                false_positives += 1
            else:
                true_negatives += 1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return precision, recall


def main():
    dir_path = "/Users/dingpengxu1/Documents/duet_大头照数据_2025_01_14_宽比_高比_面积比_分类别"
    image_data = load_image_data(dir_path)
    with_thresholds = [i * 0.01 for i in range(40, 51)]  # 0.1到1.0，步长0.1
    height_thresholds = [i * 0.01 for i in range(40, 51)]  # 0.1到1.0，步长0.1
    area_thresholds = [i * 0.01 for i in range(20, 30)]  # 0.1到1.0，步长0.1

    with open('metrics.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['width_threshold', 'height_threshold', 'area_threshold', 'precision', 'recall', 'f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        max_f1 = -1
        best_w_th = 0
        best_h_th = 0
        best_a_th = 0

        for w_th in tqdm(with_thresholds):
            for h_th in height_thresholds:
                for a_th in area_thresholds:
                    precision, recall = calculate_metrics(image_data, w_th, h_th, a_th)
                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1 = 0
                    writer.writerow({
                        'width_threshold': w_th,
                        'height_threshold': h_th,
                        'area_threshold': a_th,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    })
                    if f1 > max_f1:
                        max_f1 = f1
                        max_precision = precision
                        max_recall = recall
                        best_w_th = w_th
                        best_h_th = h_th
                        best_a_th = a_th

        print(f'最高F1分数: {max_f1:.4f} {max_precision:.4f} {max_recall:.4f}')
        print(
            f'最佳阈值组合: width_threshold={best_w_th:.2f}, height_threshold={best_h_th:.2f}, area_threshold={best_a_th:.2f}')


if __name__ == '__main__':
    main()