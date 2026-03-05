import os
import shutil
import re

# 定义源文件夹和目标文件夹
source_folder = '/Users/dingpengxu1/Documents/BlackRace'
dest_folder_less_than_027 = '/Users/dingpengxu1/Documents/BlackRace_small_27'
dest_folder_greater_or_equal_027 = '/Users/dingpengxu1/Documents/BlackRace_big_27'

# 创建目标文件夹（如果不存在）
os.makedirs(dest_folder_less_than_027, exist_ok=True)
os.makedirs(dest_folder_greater_or_equal_027, exist_ok=True)

# 正则表达式匹配亮度值
pattern = re.compile(r'brightness_([-+]?\d+\.\d+)')

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith('.jpeg') or filename.endswith('.jpg'):
        # 使用正则表达式提取亮度值
        match = pattern.search(filename)
        if match:
            brightness = float(match.group(1))  # 提取匹配的亮度值并转换为浮点数
            brightness = abs(brightness)
            print(brightness)

            # 根据亮度值决定将文件复制到哪个文件夹
            if brightness < 0.27:
                shutil.copy(os.path.join(source_folder, filename), dest_folder_less_than_027)
            else:
                shutil.copy(os.path.join(source_folder, filename), dest_folder_greater_or_equal_027)

print("图片分类完成！")