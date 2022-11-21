from typing import Tuple
from functools import partial

import torch
import torch.nn as nn

from models.uconvnext import UConvNeXt
from models.mim.MIM_block_model import DeepInfoMax


class Rec_Deep(nn.Module):
    def __init__(self,
                 out_channels: int = 3,
                 dims: Tuple[int, int, int, int] = [384, 192, 96, 3]):
        super().__init__()
        self.up_1 = UConvNeXt.Up_Basic(dim=dims[1])
        self.out_1 = nn.Conv2d(dims[1] // 2, out_channels, 1, 1)
        self.up_2 = UConvNeXt.Up_Basic(dim=dims[2])
        self.out_2 = nn.Conv2d(dims[2] // 2, out_channels, 1, 1)

    def forward(self, x):
        _, x1, x2, x3 = x
        x1 = self.out_1(self.up_1(x1))
        x2 = self.out_2(self.up_2(x2))
        return [x1, x2, x3]


class Seg_Deep(nn.Module):
    def __init__(self,
                 out_channels: int = 4,
                 dims: Tuple[int, int, int, int] = [384, 192, 96, 4]):
        super().__init__()
        self.up_1 = UConvNeXt.Up_Basic(dim=dims[1])
        self.out_1 = nn.Conv2d(dims[1] // 2, out_channels, 1, 1)
        self.up_2 = UConvNeXt.Up_Basic(dim=dims[2])
        self.out_2 = nn.Conv2d(dims[2] // 2, out_channels, 1, 1)

    def forward(self, x):
        _, x1, x2, x3 = x
        x1 = self.out_1(self.up_1(x1))
        x2 = self.out_2(self.up_2(x2))
        return [x1, x2, x3]


class Rec_Decoder(nn.Module):
    def __init__(self,
                 out_channels: int = 3,
                 feat_dims: Tuple[int, int, int, int] = [96, 192, 384, 768],
                 rec_dims: Tuple[int, int, int, int] = [384, 192, 96, 3],
                 decoder_depths: Tuple[int, int, int] = [3, 3, 3],
                 drop_path_rate: float = 0.1, ):
        super().__init__()

        self.rec_decoder_backbone = UConvNeXt.Decoder(num_class=out_channels, dims=tuple(reversed(feat_dims)),
                                                      depths=decoder_depths, drop_path_rate=drop_path_rate)
        self.rec_decoder = Rec_Deep(dims=[384, 192, 96, 3])
        self.seg_decoder = Seg_Deep(dims=[384, 192, 96, 3])

        self.mim = DeepInfoMax(0.5, 1.0, 0.1)

    def forward(self, feats, seg_feats=None):
        '''
        torch.Size([1, 384, 14, 14])
        torch.Size([1, 192, 28, 28])
        torch.Size([1, 96, 56, 56])
        torch.Size([1, 3, 224, 224])
        '''
        # 注意：在之前的MIM块中使用了 label 来帮助重构
        rec_feats = self.rec_decoder_backbone(feats)

        if seg_feats is not None:
            y = self.rec_decoder(rec_feats)
            m = self.seg_decoder(seg_feats)

            loss_mim = self.mim(m[0], m[1], m[2], y[0], y[1], y[2])
        '''
        y
        torch.Size([1, 3, 56, 56])
        torch.Size([1, 3, 112, 112])
        torch.Size([1, 3, 224, 224])
        m
        torch.Size([1, 4, 56, 56])
        torch.Size([1, 4, 112, 112])
        torch.Size([1, 4, 224, 224])
        '''
        if seg_feats is not None:
            return rec_feats[-1], loss_mim
        else:
            return rec_feats[-1], None


if __name__ == '__main__':
    '''
    Y：
    torch.Size([1, 96, 56, 56])
    torch.Size([1, 192, 28, 28])
    torch.Size([1, 384, 14, 14])
    torch.Size([1, 768, 7, 7])
    M:
    torch.Size([1, 192, 28, 28])
    torch.Size([1, 96, 56, 56])
    torch.Size([1, 4, 224, 224])
    '''
    # 重构
    x0 = torch.randn(1, 96, 56, 56)
    x1 = torch.randn(1, 192, 28, 28)
    x2 = torch.randn(1, 384, 14, 14)
    x3 = torch.randn(1, 768, 7, 7)
    # 分割
    m0 = torch.randn(1, 384, 14, 14)
    m1 = torch.randn(1, 192, 28, 28)
    m2 = torch.randn(1, 96, 56, 56)
    m3 = torch.randn(1, 4, 224, 224)

    model = Rec_Decoder()
    y_hat = model([x3, x2, x1, x0], [m0, m1, m2, m3])
    print(y_hat)
    # for i in y_hat:
    #     print(i.shape)
