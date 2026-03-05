import os
import shutil

def move_files(source_dir, target_dir):
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录中的所有文件和子目录
    for item in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item)
        target_path = os.path.join(target_dir, item)

        # 移动文件或目录
        try:
            shutil.move(source_path, target_path)
            print(f"Moved: {source_path} to {target_path}")
        except Exception as e:
            print(f"Error moving {source_path} to {target_path}: {e}")


# 设置源目录和目标目录
source_directory = '/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_10_24_9'
target_directory = '/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_10_24'

# 调用函数移动文件
move_files(source_directory, target_directory)
