import pandas as pd

# 读取CSV文件
# file_path = "/Volumes/移动硬盘1TB/duet/duet_user_emb_2024_11_30/duet_user_face_emb_final_result_2024_12_01.csv"
file_path = "/Users/dingpengxu1/Documents/duet_user_emb_2024_11_30/face_embeddings_output_part_2_2024_11_30.csv"
data = pd.read_csv(file_path)

# 计算每份的大小
num_splits = 25
split_size = len(data) // num_splits

# 分割数据
splits = [data.iloc[i * split_size: (i + 1) * split_size] for i in range(num_splits)]

# 如果有剩余的行，分配到最后一个分割中
if len(data) % num_splits != 0:
    splits[-1] = pd.concat([splits[-1], data.iloc[num_splits * split_size:]])

# 打印每份的大小
for i, split in enumerate(splits):
    print(f"Split {i+1} size: {len(split)}")

# 如果需要，可以将每份数据保存到新的CSV文件中
for i, split in enumerate(splits):
    split.to_csv(f'/Volumes/移动硬盘1TB/duet/duet_user_emb_2024_11_30/duet_face_emb_2024_12_02/duet_user_face_emb_output_split_{i+1+3}.csv', index=False)
