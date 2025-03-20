import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.freq = configs.freq
        self.channels = configs.enc_in
        self.d_model = configs.d_model
        # 年 月  日 周 时 分
        # 年趋势
        self.year_trend = nn.Parameter(torch.zeros(3, self.channels), requires_grad=True)
        self.quarter_trend = nn.Parameter(torch.zeros(4, self.channels), requires_grad=True)
        self.month_trend = nn.Parameter(torch.zeros(12, self.channels), requires_grad=True)
        self.week_trend = nn.Parameter(torch.zeros(7, self.channels), requires_grad=True)
        self.day_trend = nn.Parameter(torch.zeros(31, self.channels), requires_grad=True)
        self.hour_trend = nn.Parameter(torch.zeros(24, self.channels), requires_grad=True)
        self.day_of_year_trend = nn.Parameter(torch.zeros(366, self.channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(self.channels), requires_grad=True)
        if self.freq == 't':
            self.minute_trend = nn.Parameter(torch.zeros(4, self.channels), requires_grad=True)
        # 预测模型为一个mlp
        # self.model = nn.Sequential(
        #     nn.Linear(self.seq_len, self.d_model),
        #     nn.ReLU(),
        #     nn.Linear(self.d_model, self.pred_len)
        # )
        # self.model = nn.Linear(self.seq_len, self.pred_len)
        # self.factor = nn.Sequential(
        #     nn.Linear(self.seq_len, self.d_model),
        #     nn.ReLU(),
        #     nn.Linear(self.d_model,6))
    def get_trends(self, time_marks):
        """
        """
        # 获取所有时间特征的索引，避免多次long()操作
        time_marks = time_marks.long()  # 将所有时间标记转化为长整型
        #  B,L
        year_mark, quarter_mark, month_mark, day_mark, week_mark, hour_mark ,day_of_year_mark=\
            time_marks[:, :, 0], time_marks[:, :, 1], time_marks[:, :, 2], time_marks[:, :, 3], time_marks[:, :, 4], time_marks[:, :, 5], time_marks[:, :, 6]
        if self.freq == 't':
            minute_mark = time_marks[:, :, 6]
        # 打印时间特征的最小最大值
        # print(f"year_mark.min()={year_mark.min()}, year_mark.max()={year_mark.max()}")
        # print(f"quarter_mark.min()={quarter_mark.min()}, quarter_mark.max()={quarter_mark.max()}")
        # print(f"month_mark.min()={month_mark.min()}, month_mark.max()={month_mark.max()}")
        # print(f"week_mark.min()={week_mark.min()}, week_mark.max()={week_mark.max()}")
        # print(f"day_mark.min()={day_mark.min()}, day_mark.max()={day_mark.max()}")
        # print(f"hour_mark.min()={hour_mark.min()}, hour_mark.max()={hour_mark.max()}")

        # 获取所有对应的趋势并一次性加和
        trends = (
            self.year_trend[year_mark] +
            self.quarter_trend[quarter_mark] +
            self.month_trend[month_mark] +
            self.week_trend[week_mark] +
            self.day_trend[day_mark] +
            self.hour_trend[hour_mark]+
            self.day_of_year_trend[day_of_year_mark]
        )
        if self.freq == 't':
            trends = trends + self.minute_trend[minute_mark]

        return trends
    def forward(self, batch_x,batch_x_mark, batch_y, batch_y_mark, mask=None):
        # x_mark_enc: [Batch, Input length, 5 or 6 ] 对应年月日周时分
        # 输入数据 以及他的时间编码 96*7  预测 192*7
        # 拼接batch_x_mark和batch_y_mark，形成新的张量
        print(f"batch_x.shape={batch_x.shape}, batch_x_mark.shape={batch_x_mark.shape}, batch_y.shape={batch_y.shape}, batch_y_mark.shape={batch_y_mark.shape}")

        combined_time_marks = torch.cat((batch_x_mark, batch_y_mark), dim=1)  # (B, L_x + L_y, 6)
        print(combined_time_marks.shape)
        # 获取拼接后的时间特征的趋势
        combined_trends = self.get_trends(combined_time_marks)
        # 获取历史时间特征的趋势
        trends_x = combined_trends[:, :batch_x_mark.shape[1], :]  # (B, L_x, channels)

        # 获取未来时间特征的趋势
        trends_y = combined_trends[:, batch_x_mark.shape[1]:, :]  # (B, L_y, channels)
        # factor = self.factor(batch_x.permute(0, 2, 1)).permute(0, 2, 1)
        # 加上 bias
        trends_x = trends_x + self.bias  # (B, L_x, channels)
        trends_y = trends_y + self.bias  # (B, L_y, channels)
        # batch_x = batch_x  - trends_x  # (B, L_x, channels)
        # y_pred    = self.model(batch_x.permute(0, 2, 1)).permute(0, 2, 1) # (B, L_y, pred_len)
        # y_pred = y_pred + trends_y  # (B, L_y, pred_len)
        return trends_y
if __name__ == '__main__':
    pass







