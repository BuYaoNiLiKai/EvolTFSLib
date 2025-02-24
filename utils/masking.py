import torch

"""
B 是批次大小，L 是序列长度。
mask_shape = [B, 1, L, L] 创建一个形状为 [B, 1, L, L] 的掩码张量，其中 1 代表每个样本只有一个掩码通道，
L x L 是每个样本的掩码矩阵。
torch.triu(...) 会生成一个上三角矩阵，diagonal=1 参数指定将对角线以上部分设置为 1，剩下的部分设置为 0。
最终，self._mask 是一个形状为 [B, 1, L, L] 的布尔掩码张量，其中每一行的对角线以上的元素为 1，其他部分为 0。
"""
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

"""
ProbMask 类根据给定的得分（scores）和索引（index）生成一个基于得分的掩码。
该掩码用于某些场景中，可能根据得分或某种条件来选择哪些部分是有效的。
"""
class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
