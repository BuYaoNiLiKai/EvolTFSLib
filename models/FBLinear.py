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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_mark_enc: [Batch, Input length, 5 or 6 ] 对应年月日周时分
        x_fb = torch.sin(2*torch.pi*x_mark_enc) # [Batch, Input length, 5 or 6 ]
        x_fb = self.coefficients(x_fb) # [Batch, Input length, Channel]
        return x_fb
if __name__ == '__main__':
    pass







