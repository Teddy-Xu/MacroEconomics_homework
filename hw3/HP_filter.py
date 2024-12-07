import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter
import os

# 校验
result_dir = 'hw3/result/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 读数据（https://fred.stlouisfed.org/series/GDPC1）
data = pd.read_csv('hw3/GDPC1.csv', header=0, names=['DATE', 'GDPC1'], parse_dates=['DATE'])
data.set_index('DATE', inplace=True)

# HP滤波
# lambda参数选择1600，这是季度数据的常用值
cycle, trend = hpfilter(data['GDPC1'], lamb=1600)

# 将周期和趋势添加到数据框中
data['Trend'] = trend
data['Cycle'] = cycle

# 绘制原始数据和趋势
plt.figure(figsize=(12, 6))
plt.plot(data['GDPC1'], label='Real GDP')
plt.plot(data['Trend'], label='Trend', linestyle='--')
plt.title('U.S. Real GDP and Trend Component')
plt.legend()
plt.savefig(result_dir + 'real_gdp_and_trend.png')
plt.close()

# 绘制周期性成分
plt.figure(figsize=(12, 6))
plt.plot(data['Cycle'], label='Cyclical Component', color='orange')
plt.title('Cyclical Component of U.S. Real GDP')
plt.legend()
plt.savefig(result_dir + 'cyclical_component.png')
plt.close()

# 描述统计
desc_stats = data['Cycle'].describe()
print(desc_stats)
with open(result_dir + 'descriptive_statistics.txt', 'w') as f:
    f.write(str(desc_stats))

# 自相关图
from pandas.plotting import autocorrelation_plot

plt.figure(figsize=(8, 4))
autocorrelation_plot(data['Cycle'])
plt.title('Autocorrelation Plot of Cyclical Component')
plt.savefig(result_dir + 'autocorrelation_plot.png')
plt.close()

# 频谱分析
from scipy import signal
import numpy as np

# 计算功率谱密度
frequencies, power = signal.periodogram(data['Cycle'].dropna(), fs=4)  # 季度数据，fs=4

plt.figure(figsize=(12, 6))
plt.semilogy(frequencies, power)
plt.title('Power Spectrum of Cyclical Component')
plt.xlabel('Frequency (cycles per quarter)')
plt.ylabel('Power')
plt.savefig(result_dir + 'power_spectrum.png')
plt.close()

# 时序模型拟合（ARIMA）
from statsmodels.tsa.arima.model import ARIMA

# 拟合ARIMA模型
# 由于周期性成分是平稳的，可以尝试AR(p)模型
model = ARIMA(data['Cycle'].dropna(), order=(2, 0, 0))
results = model.fit()

summary_text = results.summary().as_text()
print(summary_text)
with open(result_dir + 'arima_summary.txt', 'w') as f:
    f.write(summary_text)

# 预测未来周期性成分
forecast_steps = 8  # 未来八个季度
forecast = results.get_forecast(steps=forecast_steps)
forecast_ci = forecast.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(data['Cycle'], label='Cyclical Component')
plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, label='Forecast', color='red')
plt.fill_between(
    forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3
)
plt.title('Forecast of Cyclical Component')
plt.legend()
plt.savefig(result_dir + 'forecast.png')
plt.close()

# 扩展：单位根检验（ADF检验）
from statsmodels.tsa.stattools import adfuller

# 对周期性成分进行ADF检验
adf_result = adfuller(data['Cycle'].dropna())
adf_output = pd.Series(
    adf_result[0:4], index=['ADF Statistic', 'p-value', '# Lags Used', '# Observations Used']
)
for key, value in adf_result[4].items():
    adf_output['Critical Value (%s)' % key] = value
print(adf_output)
with open(result_dir + 'adfuller_test.txt', 'w') as f:
    f.write(str(adf_output))

# 扩展：绘制周期性成分的直方图
plt.figure(figsize=(8, 6))
plt.hist(data['Cycle'], bins=30, edgecolor='k')
plt.title('Histogram of Cyclical Component')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig(result_dir + 'histogram.png')
plt.close()
