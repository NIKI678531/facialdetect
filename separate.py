import os
import shutil
from glob import glob
import re

def copy_first_2000_images(source_dir, target_dir, max_count=100):
    """复制指定文件夹中的前max_count张图片到目标文件夹"""
    # 创建目标文件夹（自动处理路径存在性）
    os.makedirs(target_dir, exist_ok=True)

    # 构建图片文件列表（支持常见格式）
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
    image_files = []

    # 递归获取所有匹配文件（包含大小写扩展名）
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(source_dir, '**', ext), recursive=True))
        image_files.extend(glob(os.path.join(source_dir, '**', ext.upper()), recursive=True))

    # 按文件名自然排序（保留数字顺序）
    image_files.sort(key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split('(\d+)', x)])

    # 执行复制操作
    copied_count = 0
    for idx, src_path in enumerate(image_files[:max_count], 1):
        try:
            # 构建目标路径
            filename = os.path.basename(src_path)
            dst_path = os.path.join(target_dir, filename)

            # 执行复制（保留元数据）
            shutil.copy2(src_path, dst_path)
            print(f"\rCopying: {idx}/{max_count} [{filename}]", end='', flush=True)
            copied_count += 1

        except Exception as e:
            print(f"\nError copying {src_path}: {str(e)}")
            continue

    print(f"\n\nOperation complete! Copied {copied_count} files to {target_dir}")


if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    source_folder = "/Users/lkq/Documents/duet特征数据_模型预测结果/isFourPoint/noFourPoint(five point)"
    destination_folder = "/Users/lkq/Documents/真实精确度/5"

    # 执行复制操作
    copy_first_2000_images(source_folder, destination_folder)