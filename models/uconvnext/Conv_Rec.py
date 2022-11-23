from typing import Tuple
from functools import partial

import torch
import torch.nn as nn

from models.uconvnext import UConvNeXt
from models.mim.MIM_block_model import DeepInfoMax

'''
添加功能：深度监督、MIM
先完成深度监督，在此基础上再添加MIM模块
注意：深度监督完成的是重构而非分割
'''


class Deep_sup(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Up(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x


class Rec_Decoder(nn.Module):
    def __init__(self,
                 out_channels: int = 3,
                 dims: Tuple[int, int, int, int] = [96, 192, 384, 768],
                 decoder_depths: Tuple[int, int, int] = [3, 3, 3],
                 drop_path_rate: float = 0.1, ):
        super().__init__()

        self.rec_decoder = UConvNeXt.Decoder(num_class=out_channels, dims=tuple(reversed(dims)), depths=decoder_depths,
                                             drop_path_rate=drop_path_rate)
        # self.mim = DeepInfoMax(0.5, 1.0, 0.1)

    def forward(self, y, m=None):
        y = self.rec_decoder(y)
        '''
        torch.Size([1, 384, 14, 14])
        torch.Size([1, 192, 28, 28])
        torch.Size([1, 96, 56, 56])
        torch.Size([1, 3, 224, 224])
        '''
        # loss_mim = self.mim(m[0], m[1], m[2], y[0], y[1], y[2])
        # return y, loss_mim
        return y


if __name__ == '__main__':
    '''
    torch.Size([1, 96, 56, 56])
    torch.Size([1, 192, 28, 28])
    torch.Size([1, 384, 14, 14])
    torch.Size([1, 768, 7, 7])
    '''
    # 重构
    x1 = torch.randn(1, 96, 56, 56)
    x2 = torch.randn(1, 192, 28, 28)
    x3 = torch.randn(1, 384, 14, 14)
    x4 = torch.randn(1, 768, 7, 7)
    # 分割
    m1 = torch.randn(1, 96, 56, 56)
    m2 = torch.randn(1, 192, 28, 28)
    m3 = torch.randn(1, 384, 14, 14)

    model = Rec_Decoder()
    y_hat = model([x4, x3, x2, x1], [m1, m2, m3])
    for i in y_hat:
        print(i.shape)
