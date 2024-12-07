import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter

# 读数据
data = pd.read_csv('GDPC1.csv', header=0, names=['DATE', 'GDPC1'], parse_dates=['DATE'])
data.set_index('DATE', inplace=True)

# HP滤波
# lambda参数选择1600，这是季度数据的常用值
cycle, trend = hpfilter(data['GDPC1'], lamb=1600)

# 将周期和趋势添加到数据框中
data['Trend'] = trend
data['Cycle'] = cycle

# 绘制原始数据和趋势
plt.figure(figsize=(12,6))
plt.plot(data['GDPC1'], label='实际GDP')
plt.plot(data['Trend'], label='趋势', linestyle='--')
plt.title('美国实际GDP及其趋势成分')
plt.legend()
plt.show()

# 绘制周期性成分
plt.figure(figsize=(12,6))
plt.plot(data['Cycle'], label='周期性成分', color='orange')
plt.title('美国实际GDP的周期性成分')
plt.legend()
plt.show()

# 描述统计
print(data['Cycle'].describe())

# 自相关图
from pandas.plotting import autocorrelation_plot
plt.figure(figsize=(8,4))
autocorrelation_plot(data['Cycle'])
plt.title('周期性成分的自相关图')
plt.show()

# 频谱分析（可选）
from matplotlib import mlab
plt.figure(figsize=(12,6))
plt.specgram(data['Cycle'], NFFT=256, Fs=4, noverlap=128)
plt.title('周期性成分的频谱图')
plt.xlabel('年份')
plt.ylabel('频率')
plt.show()