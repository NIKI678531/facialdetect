import logging
import os
from io import BytesIO

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from flask import Blueprint
from flask import request, jsonify

from infrastructure.models import face_detect, feature_detect
from infrastructure.utils import image_processing, clip_model
import re

# 添加这些导入
from transformers import AutoModel, AutoProcessor  # 新增导入
import torch

# 初始化部分（确保在调用processor之前执行）
# ckpt = "google/siglip2-giant-opt-patch16-384"
ckpt = "google/siglip2-so400m-patch16-384"
# device = 'cpu'
device = 'mps'
model = AutoModel.from_pretrained(ckpt).to(device).eval()  #  必须与后续输入同一设备
processor = AutoProcessor.from_pretrained(ckpt, use_fast=True)  # 关键初始化


bp = Blueprint('main', __name__)




model, preprocess = clip_model.load_siglip_model()
face_detect = face_detect.FaceDetect()
model.to(device)


class FeatureModel:
    def __init__(self, feature, directory):
        self.feature = feature
        self.model_path, self.location, self.threshold = self.get_model_weight(directory)
        self.model = self.load_model()

    def get_model_weight(self, directory):
        # 定义文件名的正则表达式模式
        pattern = re.compile(rf"{self.feature}_(\w+)_prediction_model_(\d+)_(\d+)_(\d+)\.pth")

        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                location, accuracy, recall, best_threshold = match.groups()
                model_path = os.path.join(directory, filename)
                threshold = int(best_threshold) / 100.0
                return model_path, location, threshold

        return None, None, None

    def load_model(self):
        if self.model_path is None:
            raise FileNotFoundError(f"No model found for feature {self.feature}")
        model = feature_detect.FeatureDetectNN(input_dim=512)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model

    def detect_feature(self, vector):
        print('vector', vector)
        vector = vector[0]
        vector = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(vector)
        print(output.item())
        return output.item() > self.threshold


# # 加载所有特征模型
# directory = "/Users/lkq/pycharmProject/duetFeatureClassificationPlatform/model_weight"
# # directory = "/data/model_weight"
# feature_detection_functions = {
#     #"ClearSkin": FeatureModel("ClearSkin", directory),
#     #"Smile": FeatureModel("Smile", directory),
#     "PictureHighClarity": FeatureModel("PictureHighClarity", directory),
#     # 你可以在这里添加更多的特征
# }


@bp.route('/health')
def health_check():
    # 在健康检测端点中执行一些检查，例如检查数据库连接、第三方服务等
    # 如果一切正常，返回一个成功的响应
    return 'OK', 200


def compute_person_embedding(image_input):
    try:
        if isinstance(image_input, str) and (image_input.startswith('http://') or image_input.startswith('https://')):
            # Assume it's a URL
            response = requests.get(image_input)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            # Assume it's a local image file
            image = Image.open(image_input).convert('RGB')
    except Exception as e:
        logging.error(f"Error downloading image: {str(e)}")
        return jsonify({"error": str(e)}), 400

    cropped_image = image_processing.process_image(image)
    if cropped_image is None:
        # Return a 512-dimensional zero vector if noPictureHighClarity face is detected
        # zero_vector = [np.zeros(512).tolist()]
        logging.info("No person detected, returning zero vector")
        return None
        # return zero_vector

    try:
        # ⇅ SIGLIP2 专用处理器 (修改的部分)
        inputs = processor(
            images=cropped_image,  # 使用裁剪后图像
            text=[""],  # ⇅ 必须的文本占位符
            return_tensors="pt",  # 保持PyTorch张量格式
            padding=True,  # 自动填充
            truncation=True,  # 自动截断
        ).to(model.device)  # ⇅ 设备自动同步
    except Exception as e:
        logging.error(f"Preprocessing failed: {str(e)}")
        return jsonify({"error": "Image preprocessing error"}), 500

    image_inputs = {"pixel_values": inputs["pixel_values"]}

    with torch.no_grad():
        try:
            # ⇅ SIGLIP2 特征获取方式
            image_features = model.get_image_features(** image_inputs)
            # ⇅ 保持归一化逻辑
            image_features /= image_features.norm(dim=-1, keepdim=True)
        except RuntimeError as e:
            logging.error(f"Inference error: {str(e)}")
            return jsonify({"error": "Model inference failed"}), 500

        # ⇅ 维度验证 (768维)
    if image_features.dim() != 2 or image_features.shape[1] != 768:
        logging.warning(f"Unexpected feature shape: {image_features.shape}")

    image_features = image_features.cpu().tolist()
    return image_features



def compute_face_embedding(image_input):
    try:
        if isinstance(image_input, str) and (image_input.startswith('http://') or image_input.startswith('https://')):
            # Assume it's a URL
            response = requests.get(image_input)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            # Assume it's a local image file
            image = Image.open(image_input).convert('RGB')
        image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        logging.info(f"Image downloaded from URL: {image_input}")
    except Exception as e:
        logging.error(f"Error downloading image: {str(e)}")
        return jsonify({"error": str(e)}), 400

    cropped_face = face_detect.detect_and_crop_face(image_cv2)
    if cropped_face is None:
        # Return a 512-dimensional zero vector if noPictureHighClarity face is detected
        # zero_vector = [np.zeros(512).tolist()]
        logging.info("No face detected, returning zero vector")
        # return jsonify({"face_embedding": zero_vector})
        return None

    # cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
    # image = preprocess(cropped_face_pil).unsqueeze(0)
    # image = image.to(device)
    #
    # # with torch.no_grad(), torch.cuda.amp.autocast():
    # with torch.no_grad():
    #     image_features = model.encode_image(image)
    #     image_features /= image_features.norm(dim=-1, keepdim=True)
    #     logging.info("Face embedding computed")

    # 修改后的代码：
    cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))

    # 使用SigLIP专用处理器代替原preprocess
    # 处理器会自动添加batch维度，无需手动unsqueeze(0)
    inputs = processor(
        images=cropped_face_pil,  # 输入单张图片
        return_tensors="pt",  # 返回PyTorch张量
        padding=True,  # 自动填充
        truncation=True  # 自动截断
    ).to(model.device)  # 保持与模型同一设备

    image_inputs = {"pixel_values": inputs["pixel_values"]}

    with torch.no_grad():
        # 使用get_image_features代替encode_image
        # 通过**解包处理器生成的字典参数
        image_features = model.get_image_features(** image_inputs)

        # 保持原归一化逻辑（SigLIP2需要手动归一化）
        image_features /= image_features.norm(dim=-1, keepdim=True)

    logging.info("Face embedding computed")
    print(image_features.shape)  # 预期输出 torch.Size([1, 768])
    # 实际的torch.Size([1, 1152])

    image_features = image_features.cpu().tolist()
    return image_features


def compute_image_embedding(image_input):
    print(image_input)
    try:
        if isinstance(image_input, str) and (image_input.startswith('http://') or image_input.startswith('https://')):
            # Assume it's a URL
            response = requests.get(image_input)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            # Assume it's a local image file
            image = Image.open(image_input).convert('RGB')
    except Exception as e:
        logging.error(f"Error fetching or processing image: {e}")
        return jsonify({"error": str(e)}), 400

        # 使用正确的SigLIP处理器（关键修改点）
    inputs = processor(
        images=image,  # 单张图像输入
        return_tensors="pt",  # 返回PyTorch张量
        padding=True,  # 自动填充
        truncation=True,  # 自动截断
        text=[""]  # 必须添加空文本占位符
    ).to(device)  # 确保与模型同一设备

    image_inputs = {"pixel_values": inputs["pixel_values"]}

    with torch.no_grad():
        # 使用正确的方法获取图像特征
        image_features = model.get_image_features(** image_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    image_features = image_features.cpu().tolist()
    return image_features


def compute_image_person_face_embedding(image_input):
    # 1. 计算三个嵌入向量，并优先处理错误响应
    person_emb = compute_person_embedding(image_input)
    if isinstance(person_emb, tuple):  # 如果是HTTP错误响应，直接返回
        return person_emb

    face_emb = compute_face_embedding(image_input)
    if isinstance(face_emb, tuple):
        return face_emb

    image_emb = compute_image_embedding(image_input)
    if isinstance(image_emb, tuple):
        return image_emb

    # 2. 检查是否有None（未检测到目标）
    if person_emb is None or face_emb is None or image_emb is None:
        logging.info("One or more embeddings failed (None detected)")
        return None

    # 拼接三个向量
    combined_vector = person_emb[0] + face_emb[0] + image_emb[0]
    # combined_vector = combined_vector.cpu().tolist()
    print(np.array(combined_vector).shape) #输出整个维度
    return combined_vector


def get_face_embedding(image_input):
    try:
        if isinstance(image_input, str) and (image_input.startswith('http://') or image_input.startswith('https://')):
            # Assume it's a URL
            response = requests.get(image_input)
            response.raise_for_status()
            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            # Assume it's a local image file
            # print('read image')
            img = cv2.imread(image_input)
            # print('read image success')
    except Exception as e:
        logging.error(f"Error fetching or processing image: {e}")
        return jsonify({"error": str(e)}), 400

    # print('get emb')
    face_embedding = face_detect.count_max_face_emb(img)
    # print('get emb success')

    return face_embedding


def get_face_ratio_and_eye_open(img_url):
    try:
        # 从URL读取图像
        response = requests.get(img_url)
        response.raise_for_status()
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        logging.error(f"Error fetching image: {str(e)}")
        return jsonify({"error": str(e)}), 400

    # 使用全局模型
    face_ratio, ear = face_detect.count_face_ratio_and_eye_state(img)
    logging.info("处理完成")

    return face_ratio, ear


# def detect_features(img_url, features):
#
#     if not features:
#         return jsonify({"error": "No features specified"}), 400
#
#     results = {}
#
#     for feature in features:
#         if feature in feature_detection_functions:
#             feature_model = feature_detection_functions[feature]
#             if feature_model.location == "image":
#                 # vector = image_vector
#                 vector = compute_image_embedding(img_url)
#             elif feature_model.location == "person":
#                 # vector = person_vector
#                 vector = compute_person_embedding(img_url)
#             elif feature_model.location == "face":
#                 # vector = face_vector
#                 vector = compute_face_embedding(img_url)
#             else:
#                 results[feature] = "Unknown vector type"
#                 continue
#
#             if vector is None:
#                 results[feature] = "Missing vector"
#             else:
#                 results[feature] = feature_model.detect_feature(vector)
#         else:
#             results[feature] = "Unknown feature"
#
#     return results


if __name__ == '__main__':
    img_url = 'https://cdn.duetapp.net/v1/images/eyJpZCI6IjdDWDVPUzZHRUJGSjdHWlI2QjJJU1FaREJGRDJCSSIsInciOjEyNDIsImgiOjEyNDMsImQiOjAsIm10IjoiaW1hZ2UvanBlZyIsImRoIjoxMTMzODI4MDE5Njc5MjU2Mzc3M30?format=max_480xX'

    # result = detect_features(img_url, ["ClearSkin", "Smile"])
    # print(result)
