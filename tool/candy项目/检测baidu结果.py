import pandas as pd
import os
import shutil

# 读取CSV文件
csv_file = 'duet_female_user_baidu_candy_rate_v2_add_class.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(csv_file)

# 图片所在的文件夹路径
image_folder = '/Users/dingpengxu1/Documents/duet特征数据/未打标特征数据/duet_未打标_女性性感_测试集_2025_02_11_part_1/source'  # 替换为你的图片文件夹路径
dest = '/Users/dingpengxu1/Documents/duet特征数据/未打标特征数据/百度结果_2025_02_20'
# 遍历CSV文件中的每一行
for index, row in df.iterrows():
    image_id = row['image_id']
    msg = row['msg']

    # 如果 msg 为空，则跳过
    if pd.isna(msg) or msg == '':
        print(f"跳过 {image_id}，因为 msg 为空")
        continue

    # 构建源文件路径
    src_path = os.path.join(image_folder, image_id)

    # 构建目标文件夹路径
    dest_folder = os.path.join(dest, msg)

    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 构建目标文件路径
    dest_path = os.path.join(dest_folder, image_id)

    # 移动文件
    shutil.copy(src_path, dest_path)

print("图片分类完成！")