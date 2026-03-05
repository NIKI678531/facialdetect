import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# 读取CSV文件
df = pd.read_csv('data/污染率_data.csv')

# 将日期列转换为日期格式
df['dt'] = pd.to_datetime(df['dt'])

# 将日期转换为数值（从起始日期开始的天数）
df['days'] = (df['dt'] - df['dt'].min()).dt.days

# 线性回归拟合
X = df[['days']]  # 自变量（天数）
y = df['污染率']  # 因变量（污染率）
model = LinearRegression()
model.fit(X, y)
trend_line = model.predict(X)  # 计算趋势线

# 设置全局字体样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制原始数据折线图
plt.plot(
    df['dt'],
    df['污染率'],
    marker='o',
    linestyle='-',
    color='#1f77b4',  # 原始数据颜色
    markersize=8,
    linewidth=2,
    label='Pollution Rate'  # 原始数据标签
)

# 绘制趋势线
plt.plot(
    df['dt'],
    trend_line,
    linestyle='--',
    color='#ff7f0e',  # 趋势线颜色
    linewidth=2,
    label='Trend Line'  # 趋势线标签
)

# 添加标题和标签
plt.title('Conversation Pollution Rate Over Time with Trend Line', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel('Conversation Pollution Rate', fontsize=14, fontweight='bold', labelpad=10)

# 设置日期格式
plt.gcf().autofmt_xdate()

# 显示网格（更柔和的网格）
plt.grid(True, linestyle='--', alpha=0.6)

# 添加图例
plt.legend(loc='upper right', fontsize=12)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()