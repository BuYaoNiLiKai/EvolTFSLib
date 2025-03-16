import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

def main_freq_part(x, k, rfft=True):
    # freq normalization
    # start = time.time()
    if rfft:
        xf = torch.fft.rfft(x, dim=1)
    else:
        xf = torch.fft.fft(x, dim=1)

    k_values = torch.topk(xf.abs(), k, dim=1)
    indices = k_values.indices

    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask

    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()

    norm_input = x - x_filtered
    # print(f"decompose take:{ time.time() - start} s")
    return norm_input, x_filtered
if __name__ == '__main__':
    root_path = '../dataset/'
    file_name = 'national_illness'
    print(f"file_name = {file_name}")
    day = 1/7
    file_name = root_path + file_name + '.csv'
    col = -1
    df_raw = pd.read_csv(file_name)
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
    # 转化为tensor
    target = torch.tensor(target, dtype=torch.float).unsqueeze(0).unsqueeze(-1)
    print(target.shape)
    # 进行傅里叶变换
    k=20
    xf = torch.fft.rfft(target, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0

    np.set_printoptions(suppress=True, precision=2)  # 取消科学计数法，保留两位小数
    _, top_list = torch.topk(frequency_list, k)
    print(f"top_list = {top_list}")
    top_list = top_list.detach().cpu().numpy()
    period = target.shape[1] // top_list
    # 两位小数打印
    print(f"period = {period/day}")
    period_weight = abs(xf).mean(-1)[:, top_list]
    # max min 归一化
    period_weight = (period_weight - period_weight.min()) / (period_weight.max() - period_weight.min())

    period_weight = F.softmax(period_weight, dim=1)

    print(period_weight)





