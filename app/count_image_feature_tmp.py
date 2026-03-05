import logging
import os
from io import BytesIO
import sys

# 获取当前脚本所在的目录
current_directory = os.path.dirname(os.path.abspath(__file__))

# 添加项目根目录到 sys.path
sys.path.append(os.path.join(current_directory, '..'))

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from flask import Blueprint
from flask import request, jsonify

from infrastructure.models import face_detect
from infrastructure.utils import image_processing, clip_model
import pandas as pd
import csv
from tqdm import tqdm
import concurrent.futures
import threading

bp = Blueprint('main', __name__)

# device = 'cpu'
device = 'mps'

model, preprocess = clip_model.load_siglip_model()
face_detect = face_detect.FaceDetect()
model.to(device)


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
        zero_vector = [np.zeros(512).tolist()]
        logging.info("No person detected, returning zero vector")
        # return None
        return zero_vector

    image_tensor = preprocess(cropped_image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # with torch.no_grad(), torch.cuda.amp.autocast():
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logging.info("Image features computed")
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
        zero_vector = [np.zeros(512).tolist()]
        logging.info("No face detected, returning zero vector")
        return jsonify({"face_embedding": zero_vector})
        # return None

    cropped_face_pil = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
    image = preprocess(cropped_face_pil).unsqueeze(0)
    image = image.to(device)

    # with torch.no_grad(), torch.cuda.amp.autocast():
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logging.info("Face embedding computed")

    image_features = image_features.cpu().tolist()
    return image_features


def compute_image_embedding(image_input):
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

    image = preprocess(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    image_features = image_features.cpu().tolist()
    return image_features


# 创建一个线程锁用于写入文件
file_lock = threading.Lock()


def process_row(row):
    try:
        # 获取当前行的url和user_id
        url = row['url']
        user_id = row['user_id']

        # 计算embedding
        image_emb = compute_image_embedding(url)

        # 使用锁来保证文件写入的线程安全
        with file_lock:
            with open(output_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([user_id, url, image_emb])

        return True

    except Exception as e:
        print(f"Error processing row for user_id {user_id}: {str(e)}")
        # 如果处理失败，写入空值
        with file_lock:
            with open(output_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([user_id, url, None, None, None])
        return False


def process_csv_and_save_embeddings(input_csv_path, output_csv_path):
    # 读取输入CSV文件
    df = pd.read_csv(input_csv_path)

    # 创建输出CSV文件并写入表头
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'url', 'image'])

    # 设置线程池的最大线程数
    max_workers = 10  # 可以根据需要调整

    # 使用线程池处理数据
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建future对象列表
        futures = []

        # 提交任务到线程池
        for _, row in df.iloc[244530:].iterrows():
            future = executor.submit(process_row, row)
            futures.append(future)

        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures),
                           desc="Processing images"):
            future.result()


if __name__ == '__main__':
    input_csv_path = 'user_pics.csv'
    output_csv_path = 'image_embeddings_output.csv'

    # 处理CSV文件并保存embeddings
    process_csv_and_save_embeddings(input_csv_path, output_csv_path)
