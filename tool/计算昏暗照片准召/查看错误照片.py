import os
import re
import shutil

def extract_brightness(filename):
    """从文件名中提取亮度值"""
    match = re.search(r'brightness_(\d+\.\d+)', filename)
    if match:
        return float(match.group(1))
    return None

def classify_and_copy_images(dark_folder, normal_folder, output_folder_high, output_folder_low, threshold=0.27):
    """
    根据阈值分类图片并复制到指定文件夹
    :param dark_folder: 昏暗图片文件夹
    :param normal_folder: 正常图片文件夹
    :param output_folder_high: 阈值高于0.27且不昏暗的图片输出文件夹
    :param output_folder_low: 阈值低于0.27且昏暗的图片输出文件夹
    :param threshold: 阈值，默认为0.27
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder_high, exist_ok=True)
    os.makedirs(output_folder_low, exist_ok=True)

    # 处理昏暗图片
    dark_images = [f for f in os.listdir(dark_folder) if f.endswith('.jpeg')]
    for image in dark_images:
        brightness = extract_brightness(image)
        if brightness is not None:
            if brightness > threshold:
                # 阈值低于0.27且昏暗的图片
                src_path = os.path.join(dark_folder, image)
                dst_path = os.path.join(output_folder_low, image)
                shutil.move(src_path, dst_path)
                print(f"Copied {image} to {output_folder_low} (Low Threshold & Dark)")

    # 处理正常图片
    normal_images = [f for f in os.listdir(normal_folder) if f.endswith('.jpeg')]
    for image in normal_images:
        brightness = extract_brightness(image)
        if brightness is not None:
            if brightness < threshold:
                # 阈值高于0.27且不昏暗的图片
                src_path = os.path.join(normal_folder, image)
                dst_path = os.path.join(output_folder_high, image)
                shutil.move(src_path, dst_path)
                print(f"Copied {image} to {output_folder_high} (High Threshold & Normal)")

# 文件夹路径
dark_folder = '/Users/dingpengxu1/Documents/duet_测试图片_图片昏暗_2025_01_15_带亮度/昏暗图片'
normal_folder = '/Users/dingpengxu1/Documents/duet_测试图片_图片昏暗_2025_01_15_带亮度/正常图片'

# 输出文件夹路径
output_folder_high = '/Users/dingpengxu1/Documents/duet_测试图片_图片昏暗_2025_01_15_带亮度/高于阈值_正常图片'
output_folder_low = '/Users/dingpengxu1/Documents/duet_测试图片_图片昏暗_2025_01_15_带亮度/低于阈值_昏暗图片'

# 分类并复制图片
classify_and_copy_images(dark_folder, normal_folder, output_folder_high, output_folder_low, threshold=0.27)