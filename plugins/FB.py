import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.freq = configs.freq
        self.channels = configs.enc_in
        # 年 月  日 周 时 分
        if self.freq == 'h':
            # 可学习的五个参数 初始化为0
            self.coefficients = nn.Linear(5, self.channels)
            # self.phase = nn.Parameter(torch.zeros(5, self.channels),requires_grad=True)
        elif self.freq == 't':
            self.coefficients = nn.Linear(6, self.channels)
            # self.phase = nn.Parameter(torch.zeros(6, self.channels),requires_grad=True)
        else:
            raise ValueError('Invalid frequency!')
    def loss(self, pred):
        return 0
    def normalize(self,batch_x,batch_x_mark,batch_y_mark):
        # 输入数据 以及他的时间编码 96*7  预测 192*7
        batch_x_norm = torch.sin(2*torch.pi*batch_x_mark) # [Batch, Input length, 5 or 6 ]
        batch_x_norm = self.coefficients(batch_x_norm )  # [Batch, Input length, Channel]
        batch_y_norm = torch.sin(2*torch.pi*batch_y_mark)
        batch_y_norm = self.coefficients(batch_y_norm )

        return batch_x-batch_x_norm,batch_y_norm

    def de_normalize(self, batch_x_residual,batch_x_norm):


        return batch_x_norm+batch_x_residual



