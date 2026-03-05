import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.signal as signal

# 读取CSV文件
# df = pd.read_csv('data/匹配污染率_data_2025_01_16.csv')
# df = pd.read_csv('data/滑动污染率_data_2025_01_16.csv')
df = pd.read_csv('data/聊天污染率_data_2025_01_16.csv')
df = df[['dt', '污染率']]

# 将日期列转换为日期格式
df['dt'] = pd.to_datetime(df['dt'])

# 设置全局字体样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# 创建图表
plt.figure(figsize=(12, 7))

# 将日期列设置为索引
df.set_index('dt', inplace=True)

# 按月重采样并计算平均值，使用右侧闭合区间
monthly_data = df.resample('M', closed='right', label='right').mean().reset_index()

# 处理缺失值，使用前向填充
monthly_data['污染率'].fillna(method='ffill', inplace=True)

# 应用Savitzky-Golay滤波器进行平滑处理
window_length = 12  # 滤波器长度，必须是奇数（由于按月数据点较少，减小窗口长度）
polyorder = 3      # 多项式阶数
monthly_data['savgol_smooth'] = signal.savgol_filter(monthly_data['污染率'], window_length, polyorder)

# 绘制原始数据点
# plt.plot(monthly_data['dt'], monthly_data['污染率'], marker='o', linestyle='None', color='gray', label='Original Data')

# 绘制平滑曲线
plt.plot(monthly_data['dt'], monthly_data['savgol_smooth'], linestyle='-', color='#FF7F00', label='Smoothed Curve')

# 添加标题和标签
plt.title('Monthly Average Pollution Rate Over Time', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel('Message Pollution Rate', fontsize=14, fontweight='bold', labelpad=10)

# 设置日期格式并旋转标签
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # 每月显示一个刻度
plt.xticks(rotation=45)

# 设置 y 轴范围，起点为 0
plt.ylim(0, 0.005)

# 显示网格（更柔和的网格）
# plt.grid(True, linestyle='--', alpha=0.6)

# 添加图例
# plt.legend(loc='upper right', fontsize=12)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()
