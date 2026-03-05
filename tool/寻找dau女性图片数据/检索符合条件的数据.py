import pandas as pd
import os
import shutil
from tqdm import tqdm

# 读取CSV文件
csv_path = '/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/download_duet_MAU_all_picture/duet_MAU_all_profile_picture_2025_01_12.csv'
df = pd.read_csv(csv_path)

# 按user_id降序排列
df_sorted = df.sort_values(by='user_id', ascending=False)

# 筛选出status为'default'且gender为'female'的行
filtered_df = df_sorted[(df_sorted['status'] == 'default') & (df_sorted['gender'] == 'female')]

# # 只取前30000条数据
# selected_df = filtered_df.head(30000)

# 取第30001条到第50000条数据
selected_df = filtered_df.iloc[50000:80000]

# 原始文件夹路径和目标文件夹路径
origin_folder = '/Volumes/移动硬盘1TB/duet/duet_user_picture_2025_01_13'
destination_folder = '/Users/dingpengxu1/Documents/duet特征数据/未打标特征数据/duet_未打标_女性性感_测试集_2025_02_19/source'

# 确保目标文件夹存在
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 遍历每一行，寻找图片并复制
for _, row in tqdm(selected_df.iterrows()):
    # 筛选得到的文件名
    filename = f"img_{row['user_id']}_{row['index']}.jpeg"

    # 遍历子文件夹
    for subfolder in os.listdir(origin_folder):
        subfolder_path = os.path.join(origin_folder, subfolder)
        if os.path.isdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)
            if os.path.exists(file_path):
                shutil.copy(file_path, destination_folder)
                break  # 复制成功后，跳出子文件夹循环

print("文件复制完成")
