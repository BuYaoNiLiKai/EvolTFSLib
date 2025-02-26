import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len =configs. pred_len
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
        self.fc1 = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_mark_enc: [Batch, Input length, 5 or 6 ] 对应年月日周时分

        x_fb = torch.sin(2*torch.pi*x_mark_enc) # [Batch, Input length, 5 or 6 ]
        x_fb = self.coefficients(x_fb) # [Batch, Input length, Channel]
        x_enc = x_enc-x_fb # [Batch, Input length, Channel]
        y = self.fc1(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        x_dec_mark = torch.sin(2*torch.pi*x_mark_dec) # [Batch, Prediction length, 5 or 6 ]
        x_dec_mark = self.coefficients(x_dec_mark) # [Batch, Prediction length, Channel]
        y = y+x_dec_mark
        return y
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    from data_provider.data_loader import (Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, )

    if __name__ == '__main__':
        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'custom': Dataset_Custom,
        }
        data = 'ETTm2'
        root_path = '../dataset/'
        data_path = 'ETTm2.csv'
        Data = data_dict[data]
        print(data_dict[data])
        timeenc = 0
        flag = 'train'
        shuffle_flag = False
        drop_last = False
        seq_len = 96
        label_len = 0
        pred_len = 192
        freq = 't'
        features = 'M'
        target = 'OT'
        cycle = 24
        scale = False
        fb = True
        data_set = Data(
            root_path=root_path,
            data_path=data_path,
            flag=flag,
            size=[seq_len, label_len, pred_len],
            features=features,
            target=target,
            timeenc=timeenc,
            freq=freq,
            cycle=cycle,
            scale=scale,
            fb=fb
        )
        batch_size = 32
        num_workers = 4

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=num_workers,
            drop_last=drop_last)
        print(len(data_loader))
        model = Model(seq_len, pred_len, freq, channels=7)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(data_loader):

            print(batch_x.shape, batch_y.shape, batch_x_mark.shape, batch_y_mark.shape)
            batch_x = batch_x.float()
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float()
            batch_y_mark = batch_y_mark.float()

            y = model(batch_x,batch_x_mark, batch_y,  batch_y_mark)
            print(y.shape)
            break







