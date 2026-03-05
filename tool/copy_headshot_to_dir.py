import os
import shutil
import re

# 设置阈值
width_threshold = 0.47
height_threshold = 0.43
area_threshold = 0.25

# 定义源文件夹和目标文件夹路径
src_dir = '/Users/dingpengxu1/Documents/duet_大头照数据_2025_01_14_宽比_高比_面积比'
dst_dir = '/Users/dingpengxu1/Documents/duet_大头照数据_2025_01_14_宽比_高比_面积比_大头照'
# dst_dir = '/Users/dingpengxu1/Documents/duet_大头照数据_2025_01_14_宽比_高比_面积比_非大头照'

# 确保目标文件夹存在，如果不存在则创建
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# 定义正则表达式模式来解析文件名
pattern = re.compile(r'img_\d+_\d+_w(\d+\.\d+)_h(\d+\.\d+)_a(\d+\.\d+).jpeg')

# 遍历源文件夹中的所有文件
for filename in os.listdir(src_dir):
    if filename.endswith('.jpeg'):
        # 尝试匹配文件名模式
        match = pattern.match(filename)
        if match:
            # 提取宽度、高度和面积比例
            w = float(match.group(1))
            h = float(match.group(2))
            a = float(match.group(3))
            print(w, h, a)
        else:
            # 对于不匹配的文件名，假设比例均为1.0
            w = 1.0
            h = 1.0
            a = 1.0

        # 检查是否满足条件
        # if not ((w >= width_threshold or h >= height_threshold) and a >= area_threshold):
        if ((w >= width_threshold or h >= height_threshold) and a >= area_threshold):
        # if w >= width_threshold or h >= height_threshold:
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)

            # 检查文件是否存在并复制
            if os.path.isfile(src_path):
                shutil.copy(src_path, dst_path)
                print(f'Copied {filename} to {dst_dir}')
            else:
                print(f'Error: {src_path} is not a file.')