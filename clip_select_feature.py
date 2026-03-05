import shutil
import time

import pandas as pd
import torch
import pickle

import mobileclip
import re
import os
from tqdm import tqdm

# 使用mps
device = torch.device("mps")

model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_b', pretrained='/Users/lkq/Documents/model_weight/mobileclip_blt.pt')
tokenizer = mobileclip.get_tokenizer('mobileclip_b')
model = model.to(device)  # 将模型移动到GPU

# 加载存储的特征向量
with open("/Users/lkq/Documents/dataset/duet_user_picture_2024_10_24/mobileclip_duet_image_all_features.pkl", "rb") as f:
# with open("/Users/lkq/Documents/dataset/duet_user_picture_2024_10_24/mobileclip_duet_image_face_features.pkl", "rb") as f:
# with open("/Users/lkq/Documents/dataset/duet_user_picture_2024_09_20/mobileclip_duet_image_face_features.pkl", "rb") as f:
    image_features_dict = pickle.load(f)


# 计算图片特征和文本特征相似度
def count_image_text_similarity(descriptions):
    text_tokens = tokenizer(descriptions).to(device)

    # 提取文本特征
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 计算相似度
    similarities = {filename: [] for filename in image_features_dict.keys()}
    with torch.no_grad():
        # for filename, image_features in image_features_dict.items():
        for filename, image_features in tqdm(image_features_dict.items(), desc="Calculating similarities"):
            # 将 image_features 转换为张量
            image_features = torch.tensor(image_features).to(device)
            image_features = image_features
            similarity = (text_features @ image_features.T).mean(dim=0).cpu().item()
            similarities[filename] = similarity

    return similarities


if __name__ == '__main__':
    df = pd.read_csv("/Users/lkq/Documents/dataset/duet_user_picture_2024_09_20_2024_10_24.csv")
    # df = pd.read_csv("/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_high_score_user_picture_2024_09_09_2024_09_20.csv")

    # feature = heavy digital filter causing unnatural color shifts, loss of facial detail clarity, and artificial texture. Avoid images with natural lighting or subtle retouching
    feature = "heavy digital filter causing unnatural color shifts, loss of facial detail clarity, and artificial texture. Avoid images with natural lighting or subtle retouching"
    print("feature:", feature)

    start_time = time.time()
    avg_similarities = count_image_text_similarity(feature)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"程序运行时间: {elapsed_time} 秒")
    #source_dir ='/Users/lkq/Documents/dataset/duet特征数据/已打标特征数据/duet_已打标_背景整洁情况(男)_2024_10_28/室内背景-整洁'
    source_dir = '/Users/lkq/Documents/dataset/duet_user_picture_2024_10_24/'
    # source_dir = '/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_09_20/'

    modified_feature = feature.replace(" ", "_")
    target_dir = f"/Users/lkq/Documents/duet_未打标_{modified_feature}_2025_02_24/source"

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"create {target_dir}")

    i = 0
    for filename, avg_similarity in tqdm(avg_similarities.items(), desc="copy pic"):
        if avg_similarity > 0.17:
        # 从文件名中提取user_id
            match = re.search(r'img_(\d+)_(\d+)\.jpeg', filename)
            if match:
                user_id = int(match.group(1))
                index = int(match.group(2))
                # 查找对应的性别
                user_row = df[(df['user_id'] == user_id) & (df['index'] == index)]
                # if not user_row.empty and user_row.iloc[0]['gender'] == 'male' and index == 0:
                # if not user_row.empty and user_row.iloc[0]['gender'] == 'female':
                # if not user_row.empty and index == 0:
                # 全身照或半身照
                # if not user_row.empty and user_row.iloc[0]['gender'] == 'female' and ((user_row.iloc[0]['FullBodyShot'] == 1 and user_row.iloc[0]['HalfBodyShot'] == 0) or (user_row.iloc[0]['FullBodyShot'] == 1 and user_row.iloc[0]['HalfBodyShot'] == 1 and user_row.iloc[0]['HeadShot'] == 0)):
                # if not user_row.empty and (user_row.iloc[0]['FullBodyShot'] == 1 and user_row.iloc[0]['HalfBodyShot'] == 1  ):
                if not user_row.empty:
                    if i < 10000:
                        shutil.copy2(source_dir + filename, target_dir)
                        i = i + 1

    print(f"一共有{i}张图片符合要求")
