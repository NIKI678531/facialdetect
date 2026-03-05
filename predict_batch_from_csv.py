# 从csv文件中，选出所需的特征，并把筛选结果保存到指定文件夹中
import os
import shutil

import pandas as pd

feature = 'Laidback'

# 读取csv文件
df = pd.read_csv(
    # "/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_09_20_2024_10_24.csv"
    "/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_09_07_2024_09_09.csv"
)

# source_dir = '/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_10_24/'
source_dir = '/Users/dingpengxu1/GolandProjects/qrcode_detect/tool/download_duet_user_for_feature/duet_user_picture_2024_09_09/'
target_dir = f"/Users/dingpengxu1/Documents/duet_{feature}_2024_11_25/source"

# 确保目标目录存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

############################## 基础特征 ##############################

# 男性用户
Male = df["gender"] == "male"

# 女性用户
Female = df["gender"] == "female"

# 用户分数
TemplateABC = (df["template"] == "A") | (df["template"] == "B") | (df["template"] == "C")

TemplateBC = (df["template"] == "B") | (df["template"] == "C")

TemplateA = (df["template"] == "A")

TemplateSX = (df["template"] == "S") | (df["template"] == "X")

# 全身照
FullBodyShot = (df["FullBodyShot"] == 1) & (df["HalfBodyShot"] == 0)

# 半身照
HalfBodyShot = (df["FullBodyShot"] == 1) & (df["HalfBodyShot"] == 1) & (df["HeadShot"] == 0)

# 全身照以上
AllFullBodyShot = (df["FullBodyShot"] == 1)

# 全身照 或 半身照
FullOrHalfBodyShot = (FullBodyShot | HalfBodyShot)

# 室外
Outdoor = (df["Outdoor"] == 1)

# 室内
Indoor = (df["IndoorTidy"] == 1)

# 室外整洁
OutdoorTidy = (df["Outdoor"] == 1) & (df["OutdoorTidy"] == 1)

# 室内整洁
IndoorTidy = (df["Indoor"] == 1) & (df["IndoorTidy"] == 1)

# 对镜自拍
MirrorSelfie = (df["MirrorSelfie"] == 1)

# 背景整洁
BackgroundTidy = (OutdoorTidy | IndoorTidy)

# 人和动物
PersonAnimal = (df["PersonAnimal"] == 1)

# 对镜自拍
Gym = (df["Gym"] == 1)


# 西装
MaleSuit = (df["gender"] == "male") & (df["MaleSuit"] == 1)

# 连帽衫
MaleHoodie = (df["gender"] == "male") & (df["MaleHoodie"] == 1)

# 花衬衣
MaleFloralShirt = (df["gender"] == "male") & (df["MaleFloralShirt"] == 1)

# 衬衣
MaleShirt = (df["gender"] == "male") & (df["MaleShirt"] == 1)

# 连帽衫和毛衣 男
MaleHoodieAndSweater = (df["gender"] == "male") & (df["MaleHoodieAndSweater"] == 1)

HoodieAndSweater = (df["HoodieAndSweater"] == 1)

# 短袖 男
MaleShortSleeves = (df["gender"] == "male") & (df["MaleShortSleeves"] == 1)

# 短袖
ShortSleeves = (df["ShortSleeves"] == 1)

# 女性裙子
FemaleDress = (df["gender"] == "female") & (df["FemaleDress"] == 1)

# 女性裙子
FemaleEveningDress = (df["gender"] == "female") & (df["FemaleDress"] == 1) & (df["FemaleEveningDress"] == 1)

# 眼镜
Glasses = (df["Glasses"] == 1)

# 微笑
Smile = (df["Smile"] == 1)

# 白人
WhiteRace = (df["WhiteRace"] == 1)

# 运动衫
MaleSportJersey = (df["gender"] == "male") & (df["MaleSportJersey"] == 1)

# 纯色衣服
SolidColorClothes = (df["gender"] == "male") & (df["SolidColorClothes"] == 1)

# 橄榄球
AmericanFootball = (df["AmericanFootball"] == 1)

# 足球
Soccer = (df["Soccer"] == 1)

# 篮球
Basketball = (df["Basketball"] == 1)

# 跑步
Running = (df["Running"] == 1)

# 玩乐器
PlayMusicalInstruments = (df["PlayMusicalInstruments"] == 1)

# 网球
Tennis = (df["Tennis"] == 1)

# 搏击
Boxing = (df["Boxing"] == 1)

# 棒球
BaseBall = (df["BaseBall"] == 1)

# 棒球
Golfing = (df["Golfing"] == 1)

# 摩托车
Motorcycle = (df["Motorcycle"] == 1)

# 钓鱼
Fishing = (df["Fishing"] == 1)

Sport = AmericanFootball | Soccer | Basketball | Running | Tennis | Boxing | BaseBall | Golfing

############################## 复合特征 ##############################

# sophisticated 男性 且 (全身照 或 半身照) 且 西装 且 背景整洁
# filtered_df = df[Male & FullOrHalfBodyShot & MaleSuit & BackgroundTidy]

# glamorous 女性 且 裙子
# filtered_df = df[Female & FullOrHalfBodyShot & FemaleDress & FemaleEveningDress & TemplateSX]

# glamorous 女性 且 裙子
# filtered_df = df[Male & FullOrHalfBodyShot & MaleSuit & TemplateSX]

# nerd 眼镜 且 白人
# filtered_df = df[Glasses & WhiteRace & AllFullBodyShot & TemplateABC & ~PersonAnimal & ~MaleSportJersey & ~Sport]

# Casual (全身照 或 半身照) 且 (连帽衫和毛衣 或短袖)
# filtered_df = df[FullOrHalfBodyShot & (HoodieAndSweater | ShortSleeves) & ~PersonAnimal & ~MaleSportJersey & ~Sport & ~Gym & ~Motorcycle & ~Fishing]

# Laidback (全身照 或 半身照) 且 (连帽衫和毛衣 或短袖) 且 微笑
filtered_df = df[Male & FullOrHalfBodyShot & MaleShirt & MaleFloralShirt]
# filtered_df = df[Male & FullOrHalfBodyShot & MaleFloralShirt]
# filtered_df = df[FullOrHalfBodyShot & (MaleHoodieAndSweater | MaleShortSleeves) & Smile]
# filtered_df = df[FullOrHalfBodyShot & (HoodieAndSweater | ShortSleeves) & Smile & ~PersonAnimal & ~MaleSportJersey & ~Sport & ~Gym & ~Motorcycle & ~Fishing]

# Clean cut (全身照 或 半身照) 且 纯色衣服 且 连帽衫和毛衣 且 背景整洁 且 非对镜自拍，排掉运动衫
# filtered_df = df[TemplateA & Male & FullOrHalfBodyShot & SolidColorClothes & (MaleShortSleeves | MaleHoodieAndSweater) & ~PersonAnimal & ~MaleSportJersey & ~Sport & ~Gym & ~MirrorSelfie & ~Motorcycle & ~Fishing]

# 把符合标准的的照片复制到指定路径下
for index, row in filtered_df.iterrows():
    # if index < 50000:
    image_name = f"img_{row['user_id']}_{row['index']}.jpeg"
    source_path = os.path.join(source_dir, image_name)  
    target_path = os.path.join(target_dir, image_name)

    # 检查源文件是否存在
    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)
    else:
        print(f"Image {image_name} not found.")
