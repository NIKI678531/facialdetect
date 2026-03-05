# 使用指定模型预测，并把预测结果保存到csv文件中
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
# 加载存储的特征向量
# pkl是文件名对应emb
# with open("/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_10_24/mobileclip_duet_image_all_features.pkl", "rb") as f:
with open("/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_09_09/mobileclip_duet_image_all_features.pkl", "rb") as f:
# with open("/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_09_09/mobileclip_duet_image_person_features.pkl", "rb") as f:
# with open("/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_09_09/mobileclip_duet_image_face_features.pkl", "rb") as f:
   image_features_dict = pickle.load(f)


# 读取所有的图片，把预测结果保存到csv中
def predict_and_copy(model, image_dir, feature_name, threshold):

    # 读取数据
    # df = pd.read_csv(
    #     "/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_09_20_2024_10_24.csv")
    df = pd.read_csv(
        "/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_09_07_2024_09_09.csv")

    # 初始化新列为默认值
    df[feature_name] = 0

    # 遍历所有文件
    for image_name in tqdm(os.listdir(image_dir)):
        # 从文件名中提取user_id和index
        match = re.search(r'img_(\d+)_(\d+)\.jpeg', image_name)
        if not match:
            continue

        user_id, index = map(int, match.groups())

        # 从pkl中找出文件的emb
        raw_feature = image_features_dict.get(image_name)
        if raw_feature is None:
            continue
        # 转换为torch的float32并去除多余维度
        final_feature = torch.tensor(raw_feature, dtype=torch.float32).squeeze().view(1, -1).to(device)

        # 把emb放到模型中预测
        with torch.no_grad():
            prediction = model(final_feature).item()

        # 预测结果和阈值比较，得到类别
        res = int(prediction > threshold)

        # 更新df中对应行的feature_name列
        df.loc[(df['user_id'] == user_id) & (df['index'] == index), feature_name] = res

        feature_name_score = feature_name + "_score"
        # 更新df中对应行的feature_name_score列
        df.loc[(df['user_id'] == user_id) & (df['index'] == index), feature_name_score] = prediction

    # 在df后面添加一列，把预测结果保存到csv文件中
    # df.to_csv(f"duet_user_picture_2024_09_20_2024_10_24_add_{feature_name}.csv", index=False)
    df.to_csv(f"duet_user_picture_2024_09_07_2024_09_09_add_{feature_name}.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="批量推理并复制图片")
    parser.add_argument("feature_name", type=str, help="指定特征")
    parser.add_argument("model_path", type=str, help="模型路径")
    parser.add_argument("threshold", type=float, help="预测阈值")
    # 添加一个性别的参数
    args = parser.parse_args()

    model_path = args.model_path
    # image_dir = "/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_10_24"
    image_dir = "/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_09_09"
    threshold = args.threshold
    feature_name = args.feature_name

    # 假设特征维度是512
    input_dim = 512

    model = load_model(model_path, input_dim).to(device)
    # 添加一个性别的参数
    predict_and_copy(model, image_dir, feature_name, threshold)
