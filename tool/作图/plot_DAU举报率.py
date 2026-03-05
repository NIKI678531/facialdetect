import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.nonparametric.smoothers_lowess import lowess

# 读取CSV文件
df = pd.read_csv('data/dau举报率_2025_01_16.csv')
df = df[['dt', '举报率']]

# 将日期列转换为日期格式
df['dt'] = pd.to_datetime(df['dt'], format='%Y年%m月%d日')

# 设置索引
df.set_index('dt', inplace=True)

# 处理缺失值，使用前向填充
df['举报率'].fillna(method='ffill', inplace=True)

# 去掉2024年2月15号之前的数据
df = df.loc[df.index >= '2024-02-24']

# 打印数据范围
print("Min date after filtering:", df.index.min())
print("Max date after filtering:", df.index.max())

# 准备数据
x = (df.index - df.index.min()).days
y = df['举报率']

# 应用Loess平滑
smoothed = lowess(y, x, frac=0.1)

# 将平滑后的天数转换回日期
smoothed_dates = df.index.min() + pd.to_timedelta(smoothed[:, 0], unit='D')

# 创建图表
plt.figure(figsize=(12, 7))

# 注释掉原始数据点的绘制
# plt.plot(df.index, y, marker='o', linestyle='None', color='gray', label='Original Data')

# 绘制Loess平滑曲线
plt.plot(smoothed_dates, smoothed[:, 1], linestyle='-', color='#FF7F00', label='Loess Smoothed')

# 设置日期格式并旋转标签
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)

# 添加标题和标签
plt.title('Daily Reported Rate Over Time', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel('Reported Rate', fontsize=14, fontweight='bold', labelpad=10)

plt.ylim(0.001, 0.002)

# 添加图例
# plt.legend(loc='upper right', fontsize=12)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()