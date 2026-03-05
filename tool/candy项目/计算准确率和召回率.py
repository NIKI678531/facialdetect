import os
from sklearn.metrics import precision_score, accuracy_score, recall_score

# 定义文件夹路径
# pred_candy_dir = '/Users/dingpengxu1/Documents/duet_femalesexy模型准确率计算/自有模型/candy'
# pred_non_candy_dir = '/Users/dingpengxu1/Documents/duet_femalesexy模型准确率计算/自有模型/非candy'

pred_candy_dir = '/Users/dingpengxu1/Documents/duet_femalesexy模型准确率计算/百度结果/candy'
pred_non_candy_dir = '/Users/dingpengxu1/Documents/duet_femalesexy模型准确率计算/百度结果/非candy'

true_candy_dir = '/Users/dingpengxu1/Documents/duet_femalesexy模型准确率计算/人工打标结果/candy'
true_non_candy_dir = '/Users/dingpengxu1/Documents/duet_femalesexy模型准确率计算/人工打标结果/非candy'

# 获取文件名列表
pred_candy_files = set(os.listdir(pred_candy_dir))
pred_non_candy_files = set(os.listdir(pred_non_candy_dir))
true_candy_files = set(os.listdir(true_candy_dir))
true_non_candy_files = set(os.listdir(true_non_candy_dir))

# 初始化真实标签和预测标签
y_true = []
y_pred = []

# 遍历真实标签中的文件
for file in true_candy_files:
    y_true.append(1)  # 1 表示 candy
    if file in pred_candy_files:
        y_pred.append(1)
    else:
        y_pred.append(0)  # 0 表示 非candy

for file in true_non_candy_files:
    y_true.append(0)  # 0 表示 非candy
    if file in pred_non_candy_files:
        y_pred.append(0)
    else:
        y_pred.append(1)  # 1 表示 candy

# 计算准确率和召回率
precision = precision_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f'精准度: {precision:.4f}')
print(f'召回率: {recall:.4f}')
print(f'准确率: {accuracy:.4f}')