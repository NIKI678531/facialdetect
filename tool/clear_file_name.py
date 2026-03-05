#  清理文件名
import os


def rename_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # 检查文件名是否符合指定格式
            if filename.startswith("img_") and filename.endswith(".jpeg"):
                parts_raw = filename.split('.')
                parts_handle = parts_raw[0]
                parts = parts_handle.split('_')
                if len(parts) > 2:
                    # 构建新的文件名
                    new_filename = f"{parts[0]}_{parts[1]}_{parts[2]}.jpeg"
                    # 获取完整的文件路径
                    old_file_path = os.path.join(root, filename)
                    new_file_path = os.path.join(root, new_filename)
                    # 重命名文件
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed: {old_file_path} to {new_file_path}")


# 使用你文件夹的路径替换下面的路径
# directory_path = '/Users/dingpengxu1/Documents/duet特征数据/模型预测结果/isHalfBodyShot_male_2024_10_24/noHalfBodyShot'
directory_path = '/Users/dingpengxu1/Documents/duet特征数据/已打标特征数据'
rename_files_in_directory(directory_path)
