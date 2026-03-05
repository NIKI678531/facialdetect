# 训练模型
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, mean_squared_error, f1_score, precision_score
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps"

# 将模型移动到指定设备
# model = model.to(device)


# 提取特征和标签
def extract_features_and_labels(feature, location):
    # 先检测all、person、face有没有，有的话加载，没有则跳过
    # 按照all、person、face的顺序加载
    # 加载到最后一个，加载label
    print(f"读取{feature}_image_person_features_dict")
    # 加载存储的特征向量
    with open(
            f"raw_data/is{feature}/siglip2_{feature}_image_{location}_features.pkl",
            "rb") as f:
        is_image_person_features_dict = pickle.load(f)

    print("读取noSex_image_person_features_dict")
    # 加载存储的特征向量
    with open(
            f"raw_data/is{feature}/siglip2_no{feature}_image_{location}_features.pkl",
            "rb") as f:
        no_image_person_features_dict = pickle.load(f)

    features = []
    labels = []

    # 读取Sex数据
    if isinstance(is_image_person_features_dict, dict):
        for i, (img_name, person_features) in enumerate(is_image_person_features_dict.items()):
            features.append(person_features)
            labels.append(1)

    # 读取noSex数据
    if isinstance(no_image_person_features_dict, dict):
        for i, (img_name, person_features) in enumerate(no_image_person_features_dict.items()):
            features.append(person_features)
            labels.append(0)

    return features, labels


def find_best_threshold(y_true, y_pred):
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in np.arange(0.0, 1.0, 0.01):
        y_pred_binary = (y_pred > threshold).astype(int)
        f1 = f1_score(y_true, y_pred_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)

        # 如果输入和输出维度不同，需要使用1x1卷积来匹配维度
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


# 构建和训练模型
# 构建和训练神经网络模型
def train_model(features, labels, location):
    # 将数据转换为NumPy数组
    features = np.array(features)
    labels = np.array(labels)
    print('features', features.shape)
    print('labels', labels.shape)

    # 去除多余的维度
    features = np.squeeze(features)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    print('X_train', X_train.shape)
    print('y_train', y_train.shape)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    input_dim = X_train.shape[1]
    print('input_dim', input_dim)
    model = SimpleNN(input_dim).to(device)

    # # 加载模型权重文件
    # state_dict_path = f'model_weight/is{feature}/MaleSolidShortSleeves_all_prediction_model_8080_7530_24_best_f1_model.pth'
    # state_dict = torch.load(state_dict_path)
    # model.load_state_dict(state_dict)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)

    best_f1 = 0.0
    best_model_state = None

    num_epochs = 2500
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

            model.eval()
            with torch.no_grad():
                y_pred = model(X_test.to(device)).cpu().numpy()
                y_test_binary = y_test.cpu().numpy()

                # 找到最佳阈值
                best_threshold = find_best_threshold(y_test_binary, y_pred)
                # best_threshold = 0.5
                y_pred_binary = (y_pred > best_threshold).astype(int)

                precision = precision_score(y_test_binary, y_pred_binary)
                recall = recall_score(y_test_binary, y_pred_binary)
                current_f1 = f1_score(y_test_binary, y_pred_binary)
                mse = mean_squared_error(y_test_binary, y_pred_binary)

                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Best Threshold: {best_threshold:.2f}, F1 Score: {current_f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, MSE: {mse:.4f}')

                # 更新最佳 F1 分数和模型权重
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_precision = precision
                    best_recall = recall
                    best_best_threshold = best_threshold
                    best_model_state = model.state_dict()

    directory_path = f"model_weight/is{feature}"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created.")
    else:
        print(f"Directory {directory_path} already exists.")

    if location == 'all':
        location = 'image'

    # 保存模型
    torch.save(model.state_dict(), f"model_weight/is{feature}/{feature}_{location}_prediction_model_{int(precision*10000)}_{int(recall*10000)}_{int(best_threshold*100)}_siglip2.pth")

    # 保存最佳 F1 分数的模型
    if best_model_state is not None:
        torch.save(best_model_state, f"model_weight/is{feature}/{feature}_{location}_prediction_model_{int(best_precision*10000)}_{int(best_recall*10000)}_{int(best_best_threshold*100)}_siglip2_best_f1_model.pth")

    return model


if __name__ == '__main__':
    # 读取参数
    parser = argparse.ArgumentParser(description="训练指定特征模型")
    parser.add_argument("feature", type=str, help="指定特征")
    parser.add_argument("location", type=str, help="指定部位")
    args = parser.parse_args()
    feature = args.feature
    location = args.location

    print(f"feature: {feature}")

    print("提取特征和标签")
    features, labels = extract_features_and_labels(feature, location)

    print("训练模型")
    model = train_model(features, labels, location)

    print("模型训练完成并已保存")
