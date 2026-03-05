import os
import shutil

def copy_images(source_folder, target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 检查文件是否是图片（可以根据扩展名判断）
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                # 构造源文件路径和目标文件路径
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_folder, file)

                # 直接复制文件
                shutil.copy2(source_file, target_file)
                print(f"Copied {source_file} to {target_file}")


# 使用示例
source_folder = '/Users/dingpengxu1/Documents/duet特征数据/已打标特征数据/duet_已打标_背景整洁情况(女)_2024_10_28'
target_folder = '/Users/dingpengxu1/Documents/duet特征数据/未打标特征数据/duet_未打标_照片比例(女)_2025_01_13'
copy_images(source_folder, target_folder)
