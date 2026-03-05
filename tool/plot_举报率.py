import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.signal as signal

# 读取CSV文件
df = pd.read_csv('data/分场景举报审核监控表_data.csv')
df = df[['dt', '举报率']]

# 将日期列转换为日期格式
df['dt'] = pd.to_datetime(df['dt'], format='%Y年%m月%d日')

# 设置全局字体样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# 创建图表
plt.figure(figsize=(12, 7))

# 将日期列设置为索引
df.set_index('dt', inplace=True)

# 按周重采样并计算平均值，使用右侧闭合区间
weekly_data = df.resample('W', closed='right', label='right').mean().reset_index()

# 处理缺失值，使用前向填充
weekly_data['举报率'].fillna(method='ffill', inplace=True)

# 应用Savitzky-Golay滤波器进行平滑处理
window_length = 25  # 滤波器长度，必须是奇数
polyorder = 5     # 多项式阶数
weekly_data['savgol_smooth'] = signal.savgol_filter(weekly_data['举报率'], window_length, polyorder)

# 绘制原始数据点
# plt.plot(weekly_data['dt'], weekly_data['举报率'], marker='o', linestyle='None', color='gray', label='Original Data')

# 绘制平滑曲线
plt.plot(weekly_data['dt'], weekly_data['savgol_smooth'], linestyle='-', color='#FF7F00', label='Smoothed Curve')

# 添加标题和标签
plt.title('Weekly Average Reported Rate Over Time', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel('Real Harass Reported Rate', fontsize=14, fontweight='bold', labelpad=10)

# 设置日期格式并旋转标签
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)

# 显示网格（更柔和的网格）
# plt.grid(True, linestyle='--', alpha=0.6)

# 添加图例
# plt.legend(loc='upper right', fontsize=12)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()