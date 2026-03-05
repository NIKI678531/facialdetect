import os
import random
import shutil


def random_copy_images(source_folder, target_folder, num_images):
    # 获取源文件夹下所有.jpeg图片的路径
    image_files = [os.path.join(root, file)
                   for root, dirs, files in os.walk(source_folder)
                   for file in files if file.endswith('.jpeg')]

    # 如果源文件夹中.jpeg图片数量小于要复制的数量，打印提示并返回
    if len(image_files) < num_images:
        print(f"源文件夹中.jpeg图片数量少于{num_images}张，无法完成复制操作。")
        return

    # 随机选择指定数量的图片路径
    selected_images = random.sample(image_files, num_images)

    # 复制选中的图片到目标文件夹
    for image_path in selected_images:
        shutil.copy(image_path, target_folder)


if __name__ == "__main__":
    source_folder = "/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_10_24"  # 替换为实际的源文件夹路径
    target_folder = "/Users/dingpengxu1/Documents/duet_未打标_dusky_2024_12_02/source"  # 替换为实际的目标文件夹路径
    num_images = 3000

    random_copy_images(source_folder, target_folder, num_images)