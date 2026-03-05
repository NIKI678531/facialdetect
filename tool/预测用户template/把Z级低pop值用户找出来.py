
import pandas as pd
import os
import shutil

# 读取CSV文件并创建用户信息字典
csv_path = "duet_新增男用户_2025_02_12.csv"  # 替换为您的CSV文件路径
df = pd.read_csv(csv_path, dtype={'user_id': str})
filtered_df = df[(df['template'] == 'Z') & (df['popularity'] < 0.08)]
user_popularity = filtered_df.set_index('user_id')['popularity'].to_dict()

# 设置路径
source_dir = "/Users/dingpengxu1/Documents/duet特征数据/模型预测结果/isWhiteRace/noWhiteRace"  # 替换为图片文件夹路径
target_dir = "/Users/dingpengxu1/Documents/duet特征数据/已打标特征数据/duet_已打标_非白人男性分数_2025_02_12/高分低pop用户"  # 替换为目标文件夹路径

# 确保目标文件夹存在
os.makedirs(target_dir, exist_ok=True)

# 遍历图片文件
for filename in os.listdir(source_dir):
    if filename.startswith("img_") and filename.lower().endswith(".jpeg"):
        parts = filename.split('_')

        # 验证文件名格式
        if len(parts) < 4:
            continue

        user_id = parts[1]

        # 检查用户是否符合条件
        if user_id in user_popularity:
            # 获取popularity并格式化
            popularity = user_popularity[user_id]
            popularity_str = f"{popularity:.5f}".rstrip('0').rstrip('.')

            # 构建新文件名
            new_filename = f"img_{popularity_str}_{user_id}_{parts[2]}_{'_'.join(parts[3:])}"

            # 确保扩展名正确
            if not new_filename.lower().endswith('.jpeg'):
                new_filename = new_filename.rsplit('.', 1)[0] + '.jpeg'

            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(target_dir, new_filename)
            shutil.copy2(src_path, dst_path)
            print(f"已处理: {filename} -> {new_filename}")

print("操作完成！共处理{}个文件".format(len(os.listdir(target_dir))))