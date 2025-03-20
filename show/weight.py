import numpy as np
import pandas as pd
import torch
import torch
import torch.nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from models.FBNet import Model as FBNet
class Config:
    def __init__(self, freq, enc_in):
        self.freq = freq
        self.enc_in = enc_in
        # 可以继续添加其他配置项



if __name__ == '__main__':
    configs = Config('h', 7)
    model = FBNet(configs)
    model.load_state_dict(torch.load('etth2.pth', map_location=torch.device('cpu')))
    # 加载到cpu
    root_path = '../dataset/'
    file_name = 'etth2'
    file_name = root_path + file_name + '.csv'
    model.eval()
    #获取权重
    print(model)
    weight_dict = model.state_dict()
    year_coef = weight_dict['year_trend'].detach().numpy()
    quarter_coef = weight_dict['quarter_trend'].detach().numpy()
    month_coef = weight_dict['month_trend'].detach().numpy()
    week_coef = weight_dict['week_trend'].detach().numpy()
    day_coef = weight_dict['day_trend'].detach().numpy()
    hour_coef = weight_dict['hour_trend'].detach().numpy()
    bias_coef = weight_dict['bias'].detach().numpy()
    # 获取所有时间特征的索引，避免多次long()操作
    start_time = pd.to_datetime("2016-07-01 00:00:00")
    end_time = pd.to_datetime("2018-06-26 19:00:00") # 将所有时间标记转化为长整型
    time_range = pd.date_range(start=start_time, end=end_time, freq="h")

    year_index = time_range.year - 2016
    quarter_index = (( time_range.month+ 9) % 12) // 3
    month_index = time_range.month - 1
    week_index = time_range.weekday
    day_index = time_range.day - 1
    hour_index = time_range.hour
    col = -1
    values = np.zeros_like(time_range)
    year_coef = year_coef[:, col]
    quarter_coef = quarter_coef[:, col]
    month_coef = month_coef[:, col]
    week_coef = week_coef[:, col]
    day_coef = day_coef[:, col]
    hour_coef = hour_coef[:, col]
    bias_coef = bias_coef[ col]

    values = year_coef[year_index] + quarter_coef[quarter_index] + month_coef[month_index] + week_coef[week_index] + day_coef[day_index] + hour_coef[hour_index] + bias_coef


    # 打印系数
    print(f'year_coef: {year_coef}')
    print(f'quarter_coef: {quarter_coef}')
    print(f'month_coef: {month_coef}')
    print(f'week_coef: {week_coef}')
    print(f'day_coef: {day_coef}')
    print(f'hour_coef: {hour_coef}')
    print(f'bias_coef: {bias_coef}')

    plt.plot(year_coef)
    plt.show()
    plt.plot(quarter_coef)
    plt.show()
    plt.plot(month_coef)
    plt.show()
    plt.plot(week_coef)
    plt.show()
    plt.plot(day_coef)
    plt.show()
    plt.plot(hour_coef)
    plt.show()
    # 画出真实值
    # 显示数据集

    df_raw = pd.read_csv(file_name)
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
    target = data[:, col]


    start_index = 0
    seq_len =len(time_range)
    end_index = start_index +seq_len




    plt.plot(time_range[start_index:end_index], values[start_index:end_index], label='fit values')
    # plt.plot(time_range[start_index:end_index], target[start_index:end_index], label='target')
    # plt.plot(time_range[start_index:end_index], target[start_index:end_index]-values[start_index:end_index] ,label='gap')



    plt.grid()
    plt.legend()
    plt.show()







