import os
import shutil


def organize_images(source_dir, dest_dir):
    """
    将源文件夹中的图片按模板分类到目标文件夹的对应子目录
    :param source_dir: 源图片文件夹路径
    :param dest_dir: 目标文件夹路径
    """
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)

        # 跳过子目录和非文件项
        if not os.path.isfile(file_path):
            continue

        # 分割文件名各部分
        parts = filename.split('_')

        # 验证文件名格式是否正确
        if len(parts) < 4 or not parts[0] == 'img':
            print(f"跳过文件 {filename}，文件名格式不符合要求")
            continue

        # 解析模板名称
        template = parts[2]

        # 创建目标子目录
        template_dir = os.path.join(dest_dir, template)
        os.makedirs(template_dir, exist_ok=True)

        # 构建目标路径并复制文件
        dest_path = os.path.join(template_dir, filename)
        shutil.copy(file_path, dest_path)
        print(f"已复制 {filename} => {dest_path}")


if __name__ == "__main__":
    # 设置路径（根据实际情况修改）
    SOURCE_DIR = "/Users/dingpengxu1/Documents/duet特征数据/模型预测结果/isWhiteRace/WhiteRace"
    DEST_DIR = "/Users/dingpengxu1/Documents/duet特征数据/已打标特征数据/duet_已打标_白人男性分数_2025_02_12"

    # 执行分类操作
    organize_images(SOURCE_DIR, DEST_DIR)
    print("文件分类完成！")
