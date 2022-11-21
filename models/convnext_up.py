from typing import Tuple

import torch
import torch.nn as nn
from models.backbone.convnext import convnext_base
from utils.print_model import model_structure

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock

# from models.mim.MIM_model import MIM


class ChannelAttention(nn.Module):  # Channel Attention Module
    def __init__(self, in_planes):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_ori = x
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu(max_out)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out)
        return x_ori * out


class SpatialAttention(nn.Module):  # Spatial Attention Module
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_ori = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x_ori * out


class Bottleneck(nn.Module):
    def __init__(self, channel=1024, depth=4, kernel_size=(3, 3)):
        super().__init__()
        for i in range(depth):
            dilate = 2 ** i
            model = [
                nn.Conv2d(channel, channel, kernel_size=kernel_size, padding=dilate, dilation=dilate, groups=channel),
                nn.LeakyReLU(inplace=True)]
            self.add_module('bottleneck%d' % (i + 1), nn.Sequential(*model))

    def forward(self, x):
        bottleneck_output = 0
        output = x
        for _, layer in self._modules.items():
            output = layer(output)
            bottleneck_output += output
        return bottleneck_output


class ConvNeXt_Up(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 4,
            feature_size: int = 16,
            dims: Tuple[int, int, int, int] = [128, 256, 512, 1024],

    ) -> None:
        super(ConvNeXt_Up, self).__init__()

        self.convnext = convnext_base(in_chans=in_channels)

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name='instance',
            res_block=True,
        )
        self.encoder1 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=dims[0],
            out_channels=feature_size * 2,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=False,
            res_block=True,
        )

        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=dims[1],
            out_channels=feature_size * 4,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=False,
            res_block=True,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=dims[2],
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=False,
            res_block=True,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=dims[3],
            out_channels=dims[3],
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name='instance',
            conv_block=False,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=dims[3],
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )
        self.decoder0 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name='instance',
            res_block=True,
        )

        # self.bottleneck = Bottleneck(channel=dims[-1])
        # 注意力机制
        # self.encoder1_att_c = ChannelAttention(in_planes=feature_size * 2)
        # self.encoder2_att_c = ChannelAttention(in_planes=feature_size * 4)
        # self.encoder3_att_c = ChannelAttention(in_planes=feature_size * 8)
        # self.encoder4_att_c = ChannelAttention(in_planes=dims[3])
        # self.encoder_att_s = SpatialAttention()

        self.out = UnetOutBlock(spatial_dims=2, in_channels=feature_size, out_channels=out_channels)  # type: ignore

        # MIM
        # self.MIM_model = MIM()

        self.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, train=True):
        hidden_states_out = self.convnext(x)
        '''
        x:1x3x224x224
        x1:1x128x56x56
        x2:1x256x28x28
        x3:1x512x14x14
        x4:1x1024x7x7
        '''
        x1, x2, x3, x4 = hidden_states_out

        enc0 = self.encoder0(x)  # 1x16x224x224
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x3 = self.encoder3(x3)
        x4 = self.encoder4(x4)

        # 加了注意力机制
        # x1 = self.encoder_att_s(self.encoder1_att_c(x1))
        # x2 = self.encoder_att_s(self.encoder2_att_c(x2))
        # x3 = self.encoder_att_s(self.encoder3_att_c(x3))
        # x4 = self.encoder_att_s(self.encoder4_att_c(x4))

        # 新加了Bottleneck
        # x4 = self.bottleneck(x4)

        x3 = self.decoder3(x4, x3)  # 1x128x28x28
        x2 = self.decoder2(x3, x2)  # 1x64x56x56
        x1 = self.decoder1(x2, x1)  # 1x32x112x112
        x0 = self.decoder0(x1, enc0)  # 1x16x224x224
        out = self.out(x0)

        # MIM
        # balance_loss = torch.tensor(0)
        # if train:
        #     balance_loss = self.MIM_model(x2, x1, x0, 1.0,
        #                                   x, s_label, t_data,
        #                                   x, s_label_down2, t_data_down2,
        #                                   x, s_label_down4, t_data_down4)

        # return out, x4, balance_loss
        return out, x4


if __name__ == '__main__':
    import os
    x = torch.rand(1, 3, 224, 224)
    model = ConvNeXt_Up(in_channels=3, out_channels=4)
    y_hat = model(x, train=False)
    print(y_hat[0].shape)
    print(y_hat[1].shape)

