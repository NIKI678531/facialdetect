# 使用指定模型预测，并复制预测结果到文件夹
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

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps"


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
        self.layer1 = ResidualBlock(input_dim, 512)
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


# 从图片中得到最终emb
def extract_features(image_path, location):
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

    if raw_feature is None:
        return None
    # 去除多余维度
    sq_feature = torch.tensor(raw_feature).squeeze()
    # 展平
    view_feature = sq_feature.view(1, -1)
    # 转换为numpy
    np_feature = np.array(view_feature)
    # 转换为torch的float32
    final_feature = torch.tensor(np_feature, dtype=torch.float32)
    return final_feature


def predict_and_copy(model, image_dir, output_dir, threshold, location, flag):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        try:
            feature = extract_features(image_path, location)
        except Exception as e:
            # 处理异常
            print(f"获取向量特征异常：{e}")

        if feature is not None:
            feature = feature.to(device)
            with torch.no_grad():
                prediction = model(feature).item()
            # print('prediction', prediction)

            if flag == 1:
                if prediction > threshold:
                    # 获取文件名和扩展名
                    base_name, ext = os.path.splitext(image_name)
                    # 构造新的文件名
                    new_image_name = f"{base_name}_{int(prediction * 100)}{ext}"
                    shutil.copy(image_path, os.path.join(output_dir, new_image_name))
                    print(f"Copied {new_image_name} with prediction {prediction:.4f}")
            else:
                if prediction < threshold:
                    # 获取文件名和扩展名
                    base_name, ext = os.path.splitext(image_name)
                    # 构造新的文件名
                    new_image_name = f"{base_name}_{int(prediction * 100)}{ext}"
                    # shutil.copy(image_path, os.path.join(output_dir, new_image_name))
                    print(f"Copied {new_image_name} with prediction {prediction:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="批量推理并复制图片")
    parser.add_argument("feature", type=str, help="指定特征")
    parser.add_argument("location", type=str, help="指定特征")
    parser.add_argument("model_path", type=str, help="模型路径")
    parser.add_argument("image_dir", type=str, help="图片目录")
    parser.add_argument("threshold", type=float, help="预测阈值")
    parser.add_argument("flag", type=float, help="正面还是反面")
    parser.add_argument("gender", type=str, help="性别")
    # 添加一个性别的参数
    args = parser.parse_args()

    model_path = args.model_path
    image_dir = args.image_dir
    threshold = args.threshold
    feature = args.feature
    location = args.location
    flag = args.flag

    if flag == 1:
        output_dir = f"/Users/dingpengxu1/Documents/duet特征数据/模型预测结果/is{feature}/{feature}"
    elif flag == 0:
        output_dir = f"/Users/dingpengxu1/Documents/duet特征数据/模型预测结果/is{feature}/no{feature}"

    # 假设特征维度是512
    input_dim = 512

    model = load_model(model_path, input_dim)
    # 添加一个性别的参数
    predict_and_copy(model, image_dir, output_dir, threshold, location, flag)
