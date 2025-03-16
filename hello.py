import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
# 生成时间范围
start_time = pd.to_datetime("2016-07-01 00:00:00")
end_time = pd.to_datetime("2018-06-26 19:00:00")
time_range = pd.date_range(start=start_time, end=end_time, freq="H")  # 以小时为单位

# 提取时间特征
years = (time_range.year - time_range.year.min()) / 2  # 2年周期
months = (time_range.month - 1) / 12  # 12个月周期 (0-11)
days = (time_range.day - 1) / 31  # 31天周期 (0-30)
weekdays = (time_range.weekday-1) / 7  # 7天周期 (0-6)
hours = (time_range.hour - 1) / 24  # 23小时周期 (0-23)

# 计算正弦基函数
sin_years = np.sin(2 * np.pi * years)
sin_months = np.sin(2 * np.pi * months)
sin_days = np.sin(2 * np.pi * days)
sin_weekdays = np.sin(2 * np.pi * weekdays)
sin_hours = np.sin(2 * np.pi * hours)


# 线性组合的系数 (假设系数为 [1, 0.5, 0.3, 0.2])
coeffs = np.array([1.949919   ,-0.23915228 , 0.00743314 ,-0.04592164 ,-0.08472753])
bias =-0.057460453
# coeffs = np.array([-0.15990354 ,-0.21103777 , 0.23471642  ,0.0004359 , -0.05351675])
# bias =-0.018556297
signal = coeffs[0] * sin_years + coeffs[1] * sin_months + coeffs[2] * sin_days + coeffs[3] * sin_weekdays+coeffs[4] * sin_hours+bias


# 显示数据集
root_path = './dataset/'
data_path = 'ETTh2.csv'
df_raw = pd.read_csv(os.path.join(root_path,
                                  data_path))

# 读取最后一列的数据

border1s = [0, 12 * 30 * 24 - 96, 12 * 30 * 24 + 4 * 30 * 24 - 96]
border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
border1 = border1s[0]
border2 = border2s[0]
cols_data = df_raw.columns[1:]
df_data = df_raw[cols_data]
scaler = StandardScaler()
# 转换为numpy数组
train_data = df_data[border1s[0]:border2s[0]]
scaler.fit(train_data.values)
data = scaler.transform(df_data.values)
# 获取最后一列
target = data[:, -1]

# 显示数据集

# 绘制图像
plt.figure(figsize=(12, 6))
plt.plot(time_range,target, label="Original Signal")
plt.plot(time_range, signal, label="Sine Combination Signal")  # 取前1000个点展示
plt.xlabel("Time")
plt.ylabel("Signal Value")
plt.title("Generated Periodic Signal (Years, Months, Days, Weekdays)")
plt.legend()
plt.grid()
plt.show()
