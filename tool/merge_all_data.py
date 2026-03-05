import pandas as pd
import os

# 获取当前文件夹下所有以.csv结尾的文件
directory = '/Users/dingpengxu1/Documents/duet_user_emb_2024_11_30/'
files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# 分别筛选出face、person、image相关的文件
face_files = [f for f in files if 'face_embeddings_output' in f]
person_files = [f for f in files if 'person_embeddings_output' in f]
image_files = [f for f in files if 'image_embeddings_output' in f]

print('face_files', face_files)

print("#############  开始face #############")
# 合并face文件
df_face = pd.DataFrame()
for face_file in face_files:
    print("开始：", face_file)
    file_path = os.path.join(directory, face_file)
    df_temp = pd.read_csv(file_path)
    df_face = pd.concat([df_face, df_temp])

print("#############  开始写入face #############")

df_face.to_csv('/Volumes/移动硬盘1TB/duet/duet_user_emb_2024_11_30/duet_user_face_emb_final_result_2024_12_01.csv', index=False)
# df_face = pd.read_csv('/Volumes/移动硬盘1TB/duet/duet_user_emb_2024_11_30/duet_user_face_emb_final_result_2024_12_01.csv')


# print("#############  开始person #############")
# # 合并person文件
# df_person = pd.DataFrame()
# for person_file in person_files:
#     print("开始：", person_file)
#     file_path = os.path.join(directory, person_file)
#     df_temp = pd.read_csv(file_path)
#     df_person = pd.concat([df_person, df_temp])
#
# print("#############  开始写入person #############")
# df_person.to_csv('/Volumes/移动硬盘1TB/duet/duet_user_emb_2024_11_30/duet_user_person_emb_final_result_2024_12_01.csv', index=False)
# df_person = pd.read_csv('/Volumes/移动硬盘1TB/duet/duet_user_emb_2024_11_30/duet_user_person_emb_final_result_2024_12_01.csv')

# print("#############  开始image #############")
# # 合并image文件
# df_image = pd.DataFrame()
# for image_file in image_files:
#     print("开始：", image_file)
#     file_path = os.path.join(directory, image_file)
#     df_temp = pd.read_csv(file_path)
#     df_image = pd.concat([df_image, df_temp])
#
# print("#############  开始写入image #############")
# df_image.to_csv('/Volumes/移动硬盘1TB/duet/duet_user_emb_2024_11_30/duet_user_image_emb_final_result_2024_12_01.csv', index=False)
# df_image = pd.read_csv('/Volumes/移动硬盘1TB/duet/duet_user_emb_2024_11_30/duet_user_image_emb_final_result_2024_12_01.csv')
#
# # 按照user_id和url进行内连接
#
# print("#############  连接df_image和df_person #############")
# result_df = pd.merge(df_image, df_person, on=['user_id', 'url'])
# print("#############  连接df_face #############")
# result_df = pd.merge(result_df, df_face, on=['user_id', 'url'])
#
# print("#############  开始保存 #############")
# # 保存最后的结果为新的CSV文件
# result_df.to_csv('/Volumes/移动硬盘1TB/duet/duet_user_emb_2024_11_30/duet_user_emb_final_result_2024_12_01.csv', index=False)
# print("#############  保存完成 #############")