import os


def delete_matching_files(source_folder, target_folder):
    # 获取target文件夹中的所有文件名
    target_files = set(os.listdir(target_folder))

    # 遍历source文件夹中的文件
    for filename in os.listdir(source_folder):
        # 检查文件是否在target文件夹中
        if filename in target_files:
            # 构建文件的完整路径
            file_path = os.path.join(source_folder, filename)
            # 删除文件
            os.remove(file_path)
            print(f"Deleted: {file_path}")


# 指定source和target文件夹的路径
source_folder = '/Users/dingpengxu1/PycharmProjects/duetFeatureClassificationPlatform/raw_data/isBaseBall/noBaseBall'
target_folder = '/Users/dingpengxu1/Documents/duet特征数据/已打标特征数据/duet_已打标_棒球_2024_10_30/看棒球'

# 调用函数删除匹配的文件
delete_matching_files(source_folder, target_folder)
