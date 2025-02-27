import pandas as pd
import numpy as np
import os
root_path = '../dataset/'
data_path = 'ETTm2.csv'
df_raw = pd.read_csv(os.path.join(root_path, data_path))
# 获取date
date = df_raw['date']
print(date)
#获取date的第一个元素
first_date = date[0]
start_year = int(first_date.split('-')[0])
print(start_year)
print(first_date)
# 获取他的年份
first_date = pd.to_datetime(first_date)
year = first_date.year
print(year)