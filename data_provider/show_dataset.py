import pandas as pd
from matplotlib import pyplot as plt

# 显示数据集
root_path = '../dataset/'
file_name = 'electricity'
file_name = root_path + file_name + '.csv'
data = pd.read_csv(file_name)

print(data.shape)
# 读取最后一列的数据

val_col = -1
data_col = data.iloc[:,val_col]

# 转换为numpy数组
data_col = data_col.values

# 显示数据集
plt.plot(data_col)
plt.show()



