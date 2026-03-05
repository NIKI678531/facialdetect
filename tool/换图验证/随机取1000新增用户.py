import pandas as pd

df = pd.read_csv('data/duet_batch_new_user_all_gender_2025_02_03_2025_02_04.csv')

df = df[df['user_status'] == 'default']

# 打乱数据
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# 取前1000条记录
df_first_1000 = df_shuffled.head(1000)

# 显示前5条记录以验证结果
print(df_first_1000.head())

df_first_1000.to_csv('output/duet_batch_new_user_all_gender_head_1000_2025_02_03_2025_02_04_随机2.csv', index=False)
