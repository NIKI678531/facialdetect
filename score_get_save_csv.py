import os
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import argparse
import app.count_feature as count_feature
import pickle
import torch
import argparse
import os
import shutil
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import re

device = "mps"

MODELS = [
    {
        "model_path": "/Users/lkq/pycharmProject/duetFeatureClassificationPlatform/model_weight/isFemaleFaceOne/FemaleFaceOne_person_face_all_prediction_model_6000_2903_45_siglip2_best_f1_model.pth",
        "threshold": 0.45,
        "score": 1,
        "name": "FemaleFaceOne"
    },
    {
        "model_path": "/Users/lkq/pycharmProject/duetFeatureClassificationPlatform/model_weight/isFemaleFaceTwo/FemaleFaceTwo_person_face_all_prediction_model_5833_4745_47_siglip2_best_f1_model.pth",
        "threshold": 0.47,
        "score": 2,
        "name": "FemaleFaceTwo"
    },
    {
        "model_path": "/Users/lkq/pycharmProject/duetFeatureClassificationPlatform/model_weight/isFemaleFaceThree/FemaleFaceThree_person_face_all_prediction_model_8573_9237_46_siglip2_best_f1_model.pth",
        "threshold": 0.46,
        "score": 3,
        "name": "FemaleFaceThree"
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
    # 初始化CSV数据结构
    csv_data = []

    # 创建输出目录和CSV路径
    score_dirs = {i: os.path.join(output_base_dir, f"score_{i}") for i in range(1, 5)}
    for d in score_dirs.values():
        os.makedirs(d, exist_ok=True)
    csv_path = os.path.join(output_base_dir, "prediction_records.csv")

    # 加载所有模型
    models = []
    for config in MODELS:
        model = load_model(config["model_path"], 3456)
        models.append({
            "model": model,
            "threshold": config["threshold"],
            "score": config["score"],
            "name": config["name"]
        })
    sorted_models = sorted(models, key=lambda x: x['score'], reverse=True)

    # 处理每张图片
    for image_name in tqdm(os.listdir(image_dir)):
        # 在predict_and_copy函数的循环处理部分修改record初始化
        record = {
            "filename": image_name,
            "user_id": None,
            "index": None,
            "final_score": 4,
            "error": None  # 添加默认的error字段
        }

        # 解析用户ID和索引
        match = re.search(r'img_(\d+)_(\d+)(?:_\d+)*\.jpeg', image_name)
        if match:
            record["user_id"] = int(match.group(1))
            record["index"] = int(match.group(2))

            # 初始化模型记录
            for model in MODELS:
                name = model["name"]
                record[f"{name}_score"] = None
                record[f"{name}_pred"] = 0

            try:
                # 特征提取
                image_path = os.path.join(image_dir, image_name)
                feature = extract_features(image_path, location)

                if feature is not None:
                    feature = feature.to(device)

                    # 多模型预测
                    valid_scores = []
                    for model_info in models:  # 遍历所有模型
                        with torch.no_grad():
                            prediction = model_info["model"](feature).item()

                        # 记录预测结果
                        name = model_info["name"]
                        threshold = model_info["threshold"]
                        record[f"{name}_score"] = prediction
                        record[f"{name}_pred"] = int(prediction > threshold)

                        # 收集符合条件的分数
                        if prediction > threshold:
                            valid_scores.append(model_info["score"])

                    # 确定最终分数（选择最高分）
                    if valid_scores:
                        final_score = max(valid_scores)
                    else:
                        final_score = 4
                    record["final_score"] = final_score

                    # 复制文件到对应目录
                    dest_dir = score_dirs[final_score]
                    shutil.copy(image_path, os.path.join(dest_dir, image_name))

            except Exception as e:
                print(f"处理 {image_name} 时出错: {str(e)}")
                record["error"] = str(e)

            # 添加记录到CSV数据
            csv_data.append(record)

    # 生成DataFrame并保存CSV
    df = pd.DataFrame(csv_data)


    # 同时确保columns列表包含该字段（原代码已有）
    columns = ["filename", "user_id", "index", "final_score"]
    for model in MODELS:
        name = model["name"]
        columns.extend([f"{name}_score", f"{name}_pred"])
    columns.append("error")  # 确保columns包含error

    # 调整列顺序
    columns = ["filename", "user_id", "index", "final_score"]
    for model in MODELS:
        name = model["name"]
        columns.extend([f"{name}_score", f"{name}_pred"])
    columns.append("error")

    df[columns].to_csv(csv_path, index=False)
    print(f"预测记录已保存至: {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="多分类批量推理")
    parser.add_argument("--image_dir", required=True, help="图片目录路径")
    parser.add_argument("--output_dir", required=True, help="输出目录路径")
    parser.add_argument("--location", default="person_face_all", help="特征位置")
    parser.add_argument("--gender", default="all", help="性别过滤")

    args = parser.parse_args()

    predict_and_copy(
        image_dir=args.image_dir,
        output_base_dir=args.output_dir,
        location=args.location,
        gender=args.gender
    )