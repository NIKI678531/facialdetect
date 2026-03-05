import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置API端点
endpoint = 'http://0.0.0.0:21000/compute_image_features'
headers = {'Content-Type': 'application/json'}
test_url = 'http://example.com/test_image.jpg'  # 替换为你测试用的图片URL

# 控制每秒请求数量
QPS_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 你想测试的QPS值
results = {}

def send_request():
    try:
        data = {'img_url': test_url}
        response = requests.post(endpoint, headers=headers, json=data)
        return response.status_code
    except Exception as e:
        return str(e)

def refill_semaphore(semaphore, qps):
    while True:
        time.sleep(1)
        for _ in range(qps):
            semaphore.release()

def test_qps(qps):
    semaphore = threading.Semaphore(qps)
    refill_thread = threading.Thread(target=refill_semaphore, args=(semaphore, qps), daemon=True)
    refill_thread.start()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(send_request) for _ in range(100)]  # 发送100个请求进行测试
        start_time = time.time()

        status_codes = []
        for future in as_completed(futures):
            status = future.result()
            status_codes.append(status)

        end_time = time.time()
        duration = end_time - start_time

    error_count = sum(1 for code in status_codes if code != 200)
    success_count = len(status_codes) - error_count

    return {
        'qps': qps,
        'duration': duration,
        'success_count': success_count,
        'error_count': error_count,
        'success_rate': success_count / len(status_codes) * 100
    }

for qps in QPS_list:
    print(f"Testing QPS={qps}")
    result = test_qps(qps)
    results[qps] = result
    print(result)

print("All tests completed. Results:")
for qps, result in results.items():
    print(f"QPS={qps}: {result}")
