# 给原始的数据库生成emb
from tqdm import tqdm
import os
import pickle
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
import argparse
import app.count_feature as count_feature

# 使用mps
device = torch.device("mps")

# 加载SigLIP模型和处理器
model = AutoModel.from_pretrained("google/siglip2-giant-opt-patch16-384")
processor = AutoProcessor.from_pretrained("google/siglip2-giant-opt-patch16-384")
model = model.to(device)  # 将模型移动到GPU


def process_images(image_folder, location):
    image_features_dict = {}
    for filename in tqdm(os.listdir(image_folder), desc="Processing images"):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            try:
                image_path = os.path.join(image_folder, filename)
                if location == 'person':
                    # 得到人的emb
                    raw_feature = count_feature.compute_person_embedding(image_path)
                elif location == 'face':
                    # 得到脸的emb
                    raw_feature = count_feature.compute_face_embedding(image_path)
                elif location == 'all':
                    # 得到全部的emb
                    raw_feature = count_feature.compute_image_embedding(image_path)
                elif location == 'face_only':
                    # 得到全部的emb
                    raw_feature = count_feature.get_face_embedding(image_path)
                else:
                    print(f'location err:{location}')
                    return

                if raw_feature is not None:
                    image_features_dict[filename] = raw_feature
                else:
                    print(f"Warning: {filename} returned None and will be skipped.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    return image_features_dict


def extract_emb_from_img(location):
    # base_path = "/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_10_24"
    # base_path = "/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_qianshou_user_for_feature/qianshou_female_user_picture_2024_12_09"
    base_path = "/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_female_user_picture_2024_12_08"
    # base_path = "/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_09_09"

    feature_dict = process_images(base_path, location)
    with open(os.path.join(base_path, f"mobileclip_duet_image_{location}_features.pkl"),
    # with open(os.path.join(base_path, f"mobileclip_qianshou_image_{location}_features.pkl"),
              "wb") as f:
        pickle.dump(feature_dict, f)


if __name__ == '__main__':
    # 读取参数
    parser = argparse.ArgumentParser(description="训练指定特征模型")
    # parser.add_argument("feature", type=str, help="指定特征")
    parser.add_argument("location", type=str, help="指定部位")
    args = parser.parse_args()
    # feature = args.feature
    location = args.location

    # print(f"feature: {feature}")
    extract_emb_from_img(location)
