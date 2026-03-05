import pickle
import random
import os
import numpy as np
from tqdm import tqdm

# 读取pkl文件
print("正在读取")
base_path = '/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_10_24/'
pkl_path = os.path.join(base_path, "mobileclip_duet_image_all_features.pkl")
with open(pkl_path, 'rb') as f:
    image_features_dict = pickle.load(f)

print("读取完成")
# 创建用于存储合并数据的列表
merged_data = []

# 提取所有用户ID
user_ids = set()
for filename in image_features_dict.keys():
    user_id = filename.split('_')[1]  # 提取用户ID
    user_ids.add(user_id)

# 只处理前1000个用户
user_ids = list(user_ids)[:10000]

for user_id in tqdm(user_ids):
    user_files = [fname for fname in image_features_dict if f"_{user_id}_" in fname]

    # 从用户文件中提取index为0和1的文件
    idx_0_file = next((fname for fname in user_files if fname.endswith('_0.jpeg')), None)
    idx_1_file = next((fname for fname in user_files if fname.endswith('_1.jpeg')), None)

    if idx_0_file and idx_1_file:
        emb_0 = np.array(image_features_dict[idx_0_file]).flatten()
        emb_1 = np.array(image_features_dict[idx_1_file]).flatten()

        # 随机决定哪个指标放在前面
        if random.choice([0, 1]) == 0:
            combined_emb = np.concatenate((emb_0, emb_1))  # 将索引0放在前面
            label = 0
        else:
            combined_emb = np.concatenate((emb_1, emb_0))  # 将索引1放在前面
            label = 1

        # 将结果添加到列表
        merged_data.append((user_id, combined_emb, label))

# 保存结果到pkl文件
output_path = "merged_data.pkl"
with open(output_path, 'wb') as f:
    pickle.dump(merged_data, f)

print(f"合并后的数据已保存到 {output_path}")
