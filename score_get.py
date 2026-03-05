# 使用多分类模型预测，并复制预测结果到对应分数文件夹
import pickle
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import shutil
from PIL import Image
from torchvision import transforms
import app.count_feature as count_feature
from tqdm import tqdm
import pandas as pd
import re

device = "mps"

# 预定义的三个模型配置
MODELS = [
    {
        "model_path": "/Users/lkq/pycharmProject/duetFeatureClassificationPlatform/model_weight/isFemaleFaceOne/FemaleFaceOne_person_face_all_prediction_model_6000_2903_45_siglip2_best_f1_model.pth",
        "threshold": 0.45,
        "score": 1
    },
    {
        "model_path": "/Users/lkq/pycharmProject/duetFeatureClassificationPlatform/model_weight/isFemaleFaceTwo/FemaleFaceTwo_person_face_all_prediction_model_5833_4745_47_siglip2_best_f1_model.pth",
        "threshold": 0.47,
        "score": 2
    },
    {
        "model_path": "/Users/lkq/pycharmProject/duetFeatureClassificationPlatform/model_weight/isFemaleFaceThree/FemaleFaceThree_person_face_all_prediction_model_8573_9237_46_siglip2_best_f1_model.pth",
        "threshold": 0.46,
        "score": 3
    }
]


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)

        if input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += identity
        out = self.relu(out)
        return out


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.layer1 = ResidualBlock(3456, 512)
        self.layer2 = ResidualBlock(512, 256)
        self.layer3 = ResidualBlock(256, 128)
        self.layer4 = ResidualBlock(128, 64)
        self.fc = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


def load_model(model_path, input_dim):
    model = SimpleNN(input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def extract_features(image_path, location):
    if location == 'person':
        raw_feature = count_feature.compute_person_embedding(image_path)
    elif location == 'face':
        raw_feature = count_feature.compute_face_embedding(image_path)
    elif location == 'all':
        raw_feature = count_feature.compute_image_embedding(image_path)
    elif location == 'face_only':
        raw_feature = count_feature.get_face_embedding(image_path)
    elif location == 'person_face_all':
        raw_feature = count_feature.compute_image_person_face_embedding(image_path)
    else:
        print(f'location err:{location}')
        return

    if raw_feature is None:
        return None

    sq_feature = torch.tensor(raw_feature).squeeze()
    view_feature = sq_feature.view(1, -1)
    np_feature = np.array(view_feature)
    final_feature = torch.tensor(np_feature, dtype=torch.float32)
    return final_feature


def predict_and_copy(image_dir, output_base_dir, location, gender):
    # 加载所有模型并按分数降序排列
    models = []
    for config in MODELS:
        model = load_model(config["model_path"], 3456)
        models.append({
            "model": model,
            "threshold": config["threshold"],
            "score": config["score"]
        })
    sorted_models = sorted(models, key=lambda x: x['score'], reverse=True)

    # 创建输出目录
    score_dirs = {
        1: os.path.join(output_base_dir, "score_1"),
        2: os.path.join(output_base_dir, "score_2"),
        3: os.path.join(output_base_dir, "score_3"),
        4: os.path.join(output_base_dir, "score_4")
    }
    for d in score_dirs.values():
        os.makedirs(d, exist_ok=True)

    # 这里添加你的CSV读取逻辑（如果需要）
    # df = pd.read_csv("your_csv_path.csv")

    for image_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # 性别过滤逻辑（根据实际情况修改）
        # 这里假设你已经有获取user_id和index的方法
        match = re.search(r'img_(\d+)_(\d+)(?:_\d+)*\.jpeg', image_name)
        if not match:
            continue

        # 添加你的CSV查询逻辑
        # user_id = int(match.group(1))
        # index = int(match.group(2))
        # user_row = df[...]
        # 这里简化处理，直接处理所有图片
        if gender != 'all':
            # 添加你的性别判断逻辑
            pass

        try:
            feature = extract_features(image_path, location)
        except Exception as e:
            print(f"获取向量特征异常：{e}")
            continue

        if feature is None:
            continue

        feature = feature.to(device)
        final_score = 4

        # 按分数优先级进行预测
        for model_info in sorted_models:
            with torch.no_grad():
                prediction = model_info["model"](feature).item()
            if prediction > model_info["threshold"]:
                final_score = model_info["score"]
                break

        # 复制文件到对应目录
        dest_dir = score_dirs[final_score]
        base_name = os.path.basename(image_name)
        shutil.copy(image_path, os.path.join(dest_dir, base_name))
        print(f"Copied {image_name} to score {final_score}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="多分类批量推理")
    parser.add_argument("location", type=str, help="特征位置")
    parser.add_argument("image_dir", type=str, help="图片目录")
    parser.add_argument("output_dir", type=str, help="输出根目录")
    parser.add_argument("gender", type=str, help="性别过滤")
    args = parser.parse_args()

    predict_and_copy(
        image_dir=args.image_dir,
        output_base_dir=args.output_dir,
        location=args.location,
        gender=args.gender
    )