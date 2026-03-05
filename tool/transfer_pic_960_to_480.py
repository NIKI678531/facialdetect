
from PIL import Image
import os
from tqdm import tqdm

# 源文件夹路径
source_folder = '/Users/dingpengxu1/PycharmProjects/duetFeatureClassificationPlatform/raw_data/isBackgroundToilet/noBackgroundToilet'
# 目标文件夹路径
target_folder = '/Users/dingpengxu1/PycharmProjects/duetFeatureClassificationPlatform/raw_data/isBackgroundToilet480/noBackgroundToilet'


# 创建目标文件夹（如果不存在）
os.makedirs(target_folder, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in tqdm(os.listdir(source_folder)):
    if filename.lower().endswith(('.jpg', '.jpeg')):
        # 构建完整路径
        source_path = os.path.join(source_folder, filename)
        target_path = os.path.join(target_folder, filename)

        # 打开图片
        with Image.open(source_path) as img:
            # 调整大小为480x480
            img = img.resize((480, 480))
            # 保存到目标文件夹
            img.save(target_path)