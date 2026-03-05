import pandas as pd
import os

first_file = '/Users/dingpengxu1/Documents/duet_user_emb_2024_11_30/face_embeddings_output_part_1_2024_11_30.csv'
df_first = pd.read_csv(first_file)
# 获取第一行的face列的值
first_face_value = df_first.loc[0, 'face']

face_file = '/Users/dingpengxu1/Documents/duet_user_emb_2024_11_30/face_embeddings_output_part_2_2024_11_30.csv'

# 读取CSV文件
df = pd.read_csv(face_file, usecols=['user_id', 'url', 'face'])

# 计算 'column_name' 列中空值的数量
num_missing = df['face'].isnull().sum()

print(f"空值的数量: {num_missing}")

# 使用fillna方法将face列的空值填充为第一行的face列的值
df['face'] = df['face'].fillna(first_face_value)

df.to_csv(face_file, index=False)
print(f"数据写入完成")