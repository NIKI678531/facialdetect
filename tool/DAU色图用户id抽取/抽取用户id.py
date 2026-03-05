import os
import csv

# 输入文件夹路径
folder_path = '/Users/dingpengxu1/Documents/duet特征数据/模型预测结果/isCockPicture/duet_DAU_SexPicture_part_2'  # 你需要改成你的文件夹路径

# 输出 CSV 文件路径
output_csv = 'duet_DAU_SexPicture_part_2_user_id_2024_02_08.csv'

# 获取所有文件名
file_names = os.listdir(folder_path)

# 存储结果的列表
user_ids = []

for file_name in file_names:
    # 检查文件是否是 .jpeg 文件
    if file_name.endswith('.jpeg'):
        # 拆分文件名
        parts = file_name.split('_')
        # 提取 user_id
        user_id = parts[1]
        user_ids.append(user_id)

# 去重，如果需要保留重复的 user_id，可以注释掉这一步
user_ids = list(set(user_ids))  # 去除重复的 user_id

# 保存到 CSV 文件
try:
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入标题行
        writer.writerow(['user_id'])
        # 写入数据行
        for user_id in user_ids:
            writer.writerow([user_id])
    print(f"成功保存到 {output_csv}")
except Exception as e:
    print(f"保存失败: {e}")