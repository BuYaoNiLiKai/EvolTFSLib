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
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_ = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_l.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_ = nn.Linear(self.seq_len, self.pred_len)
    def forward(self, x,x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)  # to [Batch, Channel, Input length]

        if self.individual:
            output = torch.zeros([x.size(0), x.size(1), self.pred_len],
                                          dtype=x.dtype).to(x.device)

            for i in range(self.channels):
                output[:, i, :] = self.Linear_Seasonal[i](x[:, i, :])
        else:
            output = self.Linear_Seasonal(x)


        return output.permute(0, 2, 1)  # to [Batch, Output length, Channel]
