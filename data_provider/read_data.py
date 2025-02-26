import pandas as pd
import numpy as np
import os
root_path = '../dataset/'
data_path = 'ETTm2.csv'
df_raw = pd.read_csv(os.path.join(root_path, data_path))
print(df_raw.shape)
df_stamp =df_raw[['date']]
df_stamp['date'] = pd.to_datetime(df_stamp.date)

df_stamp['month'] = df_stamp.date.apply(lambda x: x.month,1)
df_stamp['year'] = df_stamp.date.apply(lambda x: x.year,1)
df_stamp['day'] = df_stamp.date.apply(lambda x: x.day,1)
df_stamp['weekday'] = df_stamp.date.apply(lambda x: x.weekday(),1)
df_stamp['hour'] = df_stamp.date.apply(lambda x: x.hour,1)
df_stamp['minute'] = df_stamp.date.apply(lambda x: x.minute,1)
data_stamp = df_stamp.drop(['date'], axis=1).values
print(df_stamp)