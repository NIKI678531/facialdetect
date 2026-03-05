import os
from locust import HttpUser, TaskSet, task, between


class UserTasks(TaskSet):

    @task
    def get_classes_from_image(self):
        # 打开图片文件
        with open('img_1172713_0.jpeg', 'rb') as image_file:
            # 发送 POST 请求
            files = {'image': image_file}
            response = self.client.post('/get_classes_from_image', files=files)
            # 检查响应状态码
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")


class WebsiteUser(HttpUser):
    tasks = [UserTasks]
    wait_time = between(1, 2)  # 设置用户请求间隔时间