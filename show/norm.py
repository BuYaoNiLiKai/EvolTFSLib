import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

root_path = '../dataset/'
file_name = 'ETTh1'
print(f"file_name = {file_name}")
file_name = root_path + file_name + '.csv'
col = -1
df_raw = pd.read_csv(file_name)


cols_data = df_raw.columns[1:]
df_data = df_raw[cols_data]
# 转换为numpy数组
data = df_data.values
# 获取最后一列
target = data[:, col]
# 前半部分数据 减去前半部分的均值除以方差 后半部分数据 减去后半部分的均值除以方差
half_len = len(target) // 2
pre_half = (target[:half_len]-target[:half_len].mean())/target[:half_len].std()
post_half = (target[half_len:]-target[half_len:].mean())/target[half_len:].std()
# 拼接数据
# target = np.concatenate((pre_half, post_half))

# plt.plot(pre_half, label='pre_half')
# plt.plot(post_half, label='post_half')
plt.plot(pre_half-post_half, label='gap')
gap_mean = np.mean(pre_half-post_half)
print(gap_mean)
plt.legend()
plt.show()