import shutil
import time
import pandas as pd
import torch
import pickle
import mobileclip
import re
import os
from tqdm import tqdm
from PIL import Image

# 使用mps
device = torch.device("mps")

model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_b', pretrained='/Users/dingpengxu1/PycharmProjects/ml-mobileclip/model_weight/mobileclip_blt.pt')
tokenizer = mobileclip.get_tokenizer('mobileclip_b')
model = model.to(device)  # 将模型移动到GPU

# 计算图片特征
def compute_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()


# 计算图片特征和文本特征相似度
def count_image_text_similarity(image_folder, text):
    text_tokens = tokenizer(text).to(device)

    # 提取文本特征
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 计算相似度
    similarities = {}
    for filename in tqdm(os.listdir(image_folder), desc="Calculating similarities"):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(image_folder, filename)

            # if location == 'person':
            #     # 得到人的emb
            #     raw_feature = count_feature.compute_person_embedding(image_path)
            # elif location == 'face':
            #     # 得到脸的emb
            #     raw_feature = count_feature.compute_face_embedding(image_path)
            # elif location == 'all':
            #     # 得到全部的emb
            #     raw_feature = count_feature.compute_image_embedding(image_path)
            # elif location == 'person_face_all':
            #     # 得到全部的emb
            #     raw_feature = count_feature.compute_image_person_face_embedding(image_path)
            # elif location == 'face_only':
            #     # 得到全部的emb
            #     raw_feature = count_feature.get_face_embedding(image_path)
            # else:
            #     print(f'location err:{location}')
            #     return

            image_features = compute_image_embedding(image_path)
            image_features = torch.tensor(image_features).to(device)

            similarity = (text_features @ image_features.T).mean(dim=0).cpu().item()
            similarities[filename] = similarity

    return similarities


if __name__ == '__main__':
    text = "sexy woman"  # 你想要比较的文本
    source_dir = '/Volumes/移动硬盘1TB/omi/duet特征数据/未打标特征数据/duet_未打标_女性性感_2025_02_10_part_1/source'
    modified_text = text.replace(" ", "_")
    target_dir = f"/Users/dingpengxu1/Documents/duet_未打标_{modified_text}_2025_02_20/source"

    # location = 'person'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"create {target_dir}")

    start_time = time.time()
    # similarities = count_image_text_similarity(source_dir, text, location)
    similarities = count_image_text_similarity(source_dir, text)
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"程序运行时间: {elapsed_time} 秒")

    i = 0
    for filename, similarity in tqdm(similarities.items(), desc="copy pic"):
        if similarity > 0.0:  # 你可以根据需要调整相似度阈值
            # 使用正则表达式提取文件名中的各个部分
            match = re.match(r'img_(\d+)_(\d+)_(\d+)\.jpeg', filename)
            if match:
                user_id = match.group(1)
                index = match.group(2)
                score = match.group(3)
                similarity_score = f"{similarity:.4f}"  # 保留4位小数的相似度分数
                # 重新组合文件名
                new_filename = f"img_{similarity_score}_{user_id}_{index}_{score}.jpeg"
                shutil.copy2(os.path.join(source_dir, filename), os.path.join(target_dir, new_filename))
                i += 1

    print(f"一共有{i}张图片符合要求")