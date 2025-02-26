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


    def forward(self, x):
        # x: [Batch, Input length, Channel]

        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
