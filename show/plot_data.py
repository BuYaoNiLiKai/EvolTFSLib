# 显示数据集
import pandas as pd
import os

from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

root_path = '../dataset/'
data_path = 'etth1.csv'
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
start =24*240
day= 24
num_periods =5
num_day =30*12
# plt.plot(target[start:start+period], label='target')
# plt.plot(target[start+period:start+period+period], label='target')
for i in range(num_periods):
    # 进行标准化
    plot_data = target[start+i*day*num_day:start+(i+1)*day*num_day]
    plot_data = plot_data.reshape(num_day, day)
    plot_data = plot_data.mean(axis=1)
    plt.plot(plot_data, label='target'+str(i+1))
plt.show()