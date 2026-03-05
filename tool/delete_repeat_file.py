import os
import re


def delete_similar_files(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 创建一个集合来存储基础文件名（不带_99部分）
    base_files = set()

    # 正则表达式模式，用于匹配文件名
    pattern = re.compile(r'(img_\d+_\d+)(?:_\d+)?\.jpeg')

    # 遍历文件，找出基础文件名
    for file in files:
        match = pattern.match(file)
        if match:
            base_name = match.group(1) + '.jpeg'
            if file == base_name:
                base_files.add(base_name)

    # 再次遍历文件，删除不需要的文件
    for file in files:
        match = pattern.match(file)
        if match:
            base_name = match.group(1) + '.jpeg'
            if base_name in base_files and file != base_name:
                file_path = os.path.join(folder_path, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")


# 使用函数，传入你的文件夹路径
folder_path = '/Users/dingpengxu1/Documents/duet_未打标_腹肌_2024_10_25/source'
delete_similar_files(folder_path)
