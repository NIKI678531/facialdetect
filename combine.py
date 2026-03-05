import os
import shutil


def handle_conflict_files(source_dir, target_dir):
    """处理文件冲突并复制文件"""
    for item in os.listdir(source_dir):
        src_path = os.path.join(source_dir, item)

        if os.path.isfile(src_path):
            base, ext = os.path.splitext(item)
            dest_path = os.path.join(target_dir, item)
            counter = 1

            # 自动解决文件名冲突
            while os.path.exists(dest_path):
                new_filename = f"{base}_冲突{counter}{ext}"
                dest_path = os.path.join(target_dir, new_filename)
                counter += 1

            shutil.copy2(src_path, dest_path)


def merge_folders(folder1, folder2, output_folder):
    """合并两个文件夹中的同名子文件夹"""
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取两个文件夹的子目录列表
    subdirs1 = {entry.name for entry in os.scandir(folder1) if entry.is_dir()}
    subdirs2 = {entry.name for entry in os.scandir(folder2) if entry.is_dir()}

    # 找出共同子目录
    common_dirs = subdirs1 & subdirs2

    for dir_name in common_dirs:
        # 构建完整路径
        path1 = os.path.join(folder1, dir_name)
        path2 = os.path.join(folder2, dir_name)
        dest_path = os.path.join(output_folder, dir_name)

        # 创建目标目录
        os.makedirs(dest_path, exist_ok=True)

        # 合并第一个文件夹内容
        handle_conflict_files(path1, dest_path)
        # 合并第二个文件夹内容
        handle_conflict_files(path2, dest_path)

        print(f"已合并目录: {dir_name}")

    print("\n合并完成！")
    print(f"总合并子目录数: {len(common_dirs)}")
    print(f"输出位置: {os.path.abspath(output_folder)}")


if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    SOURCE_FOLDER1 = "/Users/lkq/Downloads/照片质量男性照片已分类等级1-5"
    SOURCE_FOLDER2 = "/Users/lkq/Downloads/照片质量女性用户图片分类等级1-5"
    OUTPUT_FOLDER = "/Users/lkq/Documents/照片质量男和女"

    # 执行合并
    merge_folders(SOURCE_FOLDER1, SOURCE_FOLDER2, OUTPUT_FOLDER)
