import os
import shutil

# 定义文件夹路径
pred_candy_dir = '/Users/dingpengxu1/Documents/duet_femalesexy模型准确率计算/百度结果/candy'
true_non_candy_dir = '/Users/dingpengxu1/Documents/duet_femalesexy模型准确率计算/人工打标结果/非candy'
misclassified_dir = '/Users/dingpengxu1/Documents/duet_femalesexy模型准确率计算/人工打标结果/误判文件'

# 创建误判文件夹（如果不存在）
if not os.path.exists(misclassified_dir):
    os.makedirs(misclassified_dir)

# 获取文件名列表
pred_candy_files = set(os.listdir(pred_candy_dir))
true_non_candy_files = set(os.listdir(true_non_candy_dir))

# 找到模型预测为candy但真实标签为非candy的文件
misclassified_files = pred_candy_files.intersection(true_non_candy_files)

# 移动这些文件到误判文件夹
for file in misclassified_files:
    src_path = os.path.join(true_non_candy_dir, file)
    dst_path = os.path.join(misclassified_dir, file)
    shutil.move(src_path, dst_path)
    print(f'Moved: {file}')

print(f'Total misclassified files moved: {len(misclassified_files)}')