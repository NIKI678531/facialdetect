import os
import shutil

# 定义源文件夹和目标文件夹
source1 = '/Users/dingpengxu1/Documents/duet特征数据/未打标特征数据/duet_未打标_女性性感_测试集_2025_02_11_part_1/source'
source2 = '/Users/dingpengxu1/Documents/duet_femalesexy模型准确率计算/百度结果/candy'
source3 = '/Users/dingpengxu1/Documents/duet_femalesexy模型准确率计算/百度结果/非candy'

# 确保source3文件夹存在
if not os.path.exists(source3):
    os.makedirs(source3)

# 遍历source1文件夹中的所有文件
for filename in os.listdir(source1):
    # 构建source1和source2中的文件路径
    file_path_source1 = os.path.join(source1, filename)
    file_path_source2 = os.path.join(source2, filename)

    # 检查source2中是否存在该文件
    if not os.path.exists(file_path_source2):
        # 如果不存在，则复制到source3
        shutil.copy(file_path_source1, source3)
        print(f"Copied {filename} to {source3}")