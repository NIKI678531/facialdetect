import os
import shutil
import re


def rename_and_copy_files(source_folder, destination_folder):
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 获取源文件夹中的所有文件
    files = os.listdir(source_folder)

    # 正则表达式匹配符合条件的文件名
    pattern = re.compile(r'img_(\d+)_(\d+)_w\d+\.\d+_h\d+\.\d+_a(\d+\.\d+)\.jpeg')

    for file in files:
        # 如果文件是符合条件的文件
        match = pattern.match(file)
        if match:
            part1, part2, area = match.groups()
            parts = file.split('_')
            # 提取宽和高的参数
            width = parts[3]
            height = parts[4]
            # 重命名文件
            new_name = f'img_a{area}_{part1}_{part2}_{width}_{height}.jpeg'
        else:
            # 如果文件名不符合格式，则保留原始文件名
            new_name = file

        # 生成源文件和目标文件的完整路径
        src_file_path = os.path.join(source_folder, file)
        dst_file_path = os.path.join(destination_folder, new_name)

        # 复制并重命名文件
        shutil.copy(src_file_path, dst_file_path)
        print(f'Copied and renamed {file} to {new_name}')


# 使用示例
source_folder = "/Users/dingpengxu1/Documents/duet用户大头照阈值确定/duet_随机1000个新增用户主图_带脸部占比_2025_02_06"  # 替换为你的源文件夹路径
destination_folder = "/Users/dingpengxu1/Documents/duet用户大头照阈值确定/duet_随机1000个新增用户主图_带脸部占比_按人脸占比排序_2025_02_06"  # 替换为你的目标文件夹路径

rename_and_copy_files(source_folder, destination_folder)
