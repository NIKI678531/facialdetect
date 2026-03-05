import os
import re

# 文件夹路径
folder_path = '/Users/dingpengxu1/Documents/duet_femalesexy模型准确率计算/自有模型/性感'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 使用正则表达式匹配文件名
    match = re.match(r'img_(\d+)_(\d+)_(\d+)(?:_\d+)*\.jpeg', filename)
    if match:
        # 提取 user_id, index, score1
        user_id, index, score1 = match.groups()[:3]
        # 构建新的文件名
        new_filename = f'img_{user_id}_{index}_{score1}.jpeg'
        # 获取文件的完整路径
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)
        # 重命名文件
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} -> {new_filename}')