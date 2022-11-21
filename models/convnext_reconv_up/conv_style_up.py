import torch
import torch.nn as nn

from typing import Tuple

from models.backbone.convnext import convnext_base
# from ConvNeXt_UP_style.style_transfer import Style_Transfer
from models.convnext_reconv_up.up_convnext import Reverse_ConvNeXt


class Bottleneck(nn.Module):
    def __init__(self, channel):
        super(Bottleneck, self).__init__()
        self.group_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.point_conv = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self, x):
        return self.point_conv(self.relu(self.bn(self.group_conv(x))))


class ConvNeXt_Up(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 4,
            feature_size: int = 16,
            dims: Tuple[int, int, int, int] = [128, 256, 512, 1024],

    ) -> None:
        super(ConvNeXt_Up, self).__init__()

        self.convnext = convnext_base(in_chans=in_channels, depths=[2, 2, 2, 2])

        # self.Bottleneck = Bottleneck(1024 * 2)
        # self.mean_std_block = Style_Transfer(1024 * 2)

        # up
        self.up_convnext = Reverse_ConvNeXt()

    def forward(self, source, target=None, train=True):
        # encoder_convnext
        '''
        1x128x56x56
        1x256x28x28
        1x512x14x14
        1x1024x7x7
        '''
        s_hidden_states = self.convnext(source)
        x1, x2, x3, x4 = s_hidden_states
        # t_hidden_states = self.convnext(target)

        # style_transfer
        # t_mean_std = self.mean_std_block(s_hidden_states[-1], t_hidden_states[-1])

        t_out = self.up_convnext([x4, x3, x2, x1])
        return t_out, x4


if __name__ == '__main__':
    source = torch.randn(1, 3, 224, 224)
    model = ConvNeXt_Up()
    y_hat = model(source)
    # print(y_hat.shape)
