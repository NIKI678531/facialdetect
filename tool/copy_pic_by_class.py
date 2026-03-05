import os
import shutil

# 定义文件夹路径
dir1 = '/Users/dingpengxu1/Documents/duet_大头照数据_2025_01_14_宽比_高比_面积比'  # 替换为dir1的实际路径
dir2 = '/Users/dingpengxu1/Documents/duet特征数据/已打标特征数据/duet_已打标_大头照_2025_01_15'  # 替换为dir2的实际路径
dir3 = '/Users/dingpengxu1/Documents/duet_大头照数据_2025_01_14_宽比_高比_面积比_分类别'  # 替换为dir3的实际路径

# 创建dir3及其子文件夹
os.makedirs(os.path.join(dir3, '大头照'), exist_ok=True)
os.makedirs(os.path.join(dir3, '非大头照'), exist_ok=True)
os.makedirs(os.path.join(dir3, '其他'), exist_ok=True)

# 遍历dir2中的子文件夹
for subdir in ['大头照', '非大头照', '其他']:
    subdir_path = os.path.join(dir2, subdir)

    # 遍历子文件夹中的图片
    for filename in os.listdir(subdir_path):
        if filename.endswith('.jpeg'):
            # 提取图片名中的基础部分（去掉后缀）
            base_name = os.path.splitext(filename)[0]

            # 在dir1中查找匹配的图片
            for dir1_filename in os.listdir(dir1):
                if dir1_filename.startswith(base_name) and dir1_filename.endswith('.jpeg'):
                    # 复制图片到dir3中的相应子文件夹
                    src_path = os.path.join(dir1, dir1_filename)
                    dst_path = os.path.join(dir3, subdir, dir1_filename)
                    shutil.copy(src_path, dst_path)
                    print(f'Copied {src_path} to {dst_path}')

print('Done!')