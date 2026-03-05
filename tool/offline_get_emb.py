import pandas as pd
import requests
import json
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

# 读取CSV文件
df = pd.read_csv('user_pics.csv')

# 添加新列
df['image'] = None
df['person'] = None
df['face'] = None

# API端点和对应的返回键名
endpoints = {
    'image': ('http://0.0.0.0:21000/compute_image_features', 'image_features'),
    'person': ('http://0.0.0.0:21000/compute_image_person_features', 'image_features'),
    'face': ('http://0.0.0.0:21000/compute_image_face_features', 'face_embedding')
}

# 请求头
headers = {'Content-Type': 'application/json'}

# 控制每秒请求数量
QPS = 5  # 每秒请求数
semaphore = Semaphore(QPS)

def fetch_features(index, url, feature_type, endpoint, response_key):
    with semaphore:  # 使用信号量控制请求速率
        try:
            # 准备请求数据
            data = {'img_url': url}

            # 发送请求
            response = requests.post(endpoint, headers=headers, json=data)

            if response.status_code == 200:
                # 提取特征向量
                features = response.json()[response_key][0]
                # 将特征向量转换为字符串
                features_str = ','.join(map(str, features))
                return index, feature_type, features_str
            else:
                print(f"Error status code {response.status_code} for {url} at {feature_type}")
                return index, feature_type, 'error'

        except Exception as e:
            print(f"Error processing {url} for {feature_type}: {str(e)}")
            return index, feature_type, 'error'

def refill_semaphore():
    while True:
        time.sleep(1)
        for _ in range(QPS):
            semaphore.release()

# 启动一个线程来定期补充信号量
import threading
refill_thread = threading.Thread(target=refill_semaphore, daemon=True)
refill_thread.start()

# 使用ThreadPoolExecutor进行多线程请求
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    i = 0
    for index in tqdm(df.index, desc="Processing rows"):
        if i < 100:
            continue
        row = df.loc[index]
        url = row['url']
        for feature_type, (endpoint, response_key) in endpoints.items():
            futures.append(executor.submit(fetch_features, index, url, feature_type, endpoint, response_key))

    for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching features"):
        index, feature_type, result = future.result()
        df.at[index, feature_type] = result

    # 每处理完一行就保存一次
    df.to_csv('user_pics_with_features.csv', index=False)

print("Processing completed!")
