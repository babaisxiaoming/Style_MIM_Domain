import torch
import torch.nn as nn
from models.ConvNeXt_UNETR.convnext import convnext_base
from models.other.convnext_uper_model.decoder import FPNHEAD
import torchinfo
from models.mim.MIM_model import MIM


def summary_model(model):
    torchinfo.summary(model, (3, 224, 224), batch_dim=0,
                      col_names=('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'), verbose=1)


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


class ConvNeXt_Uper(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXt_Uper, self).__init__()
        self.num_classes = num_classes
        self.backbone = convnext_base(in_chans=3)
        self.in_channels = 1024
        self.channels = 256
        self.decoder = FPNHEAD(channels=self.in_channels, out_channels=self.channels)
        self.cls_seg = nn.Sequential(
            nn.Conv2d(256, self.num_classes, kernel_size=3, padding=1),
        )

        # bottleneck
        self.bottleneck = Bottleneck(channel=self.in_channels)

        # MIM
        self.MIM_model = MIM()

    def forward(self, x, train=True,
                s_label=None, t_data=None, s_label_down2=None, t_data_down2=None,
                s_label_down4=None, t_data_down4=None):
        x_ori = x
        '''
        x:  1x3x224x224
        x1: 1x128x56x56
        x2: 1x256x28x28
        x3: 1x512x14x14
        x4: 1x1024x7x7
        '''
        x1, x2, x3, x4 = self.backbone(x)

        # bottleneck
        x_bottleneck = self.bottleneck(x4)

        # x1,x2,x3是为了MIM块
        x, x3, x2, x1 = self.decoder([x1, x2, x3, x4])

        x = nn.functional.interpolate(x, size=(x.size(2) * 4, x.size(3) * 4), mode='bilinear', align_corners=True)
        # get_MIM
        # x3 = x3
        x2 = nn.functional.interpolate(x2, size=(x2.size(2) * 2, x2.size(3) * 2), mode='bilinear', align_corners=True)
        x1 = nn.functional.interpolate(x1, size=(x1.size(2) * 4, x1.size(3) * 4), mode='bilinear', align_corners=True)

        out = self.cls_seg(x)
        # return x, x1, x2, x3

        # MIM
        balance_loss = 0.
        if train:
            balance_loss = self.MIM_model(x3, x2, x1, 1.0,
                                          x_ori, s_label, t_data,
                                          x_ori, s_label_down2, t_data_down2,
                                          x_ori, s_label_down4, t_data_down4)

        return out, x_bottleneck, balance_loss


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    # InstanceNorm2d、LeakyReLU、ConvTranspose2d替换
    x = torch.randn(1, 3, 224, 224).cuda()
    model = ConvNeXt_Uper(num_classes=4).cuda()
    y_hat = model(x)
    # for i in y_hat:
    #     print(i.shape)

    summary_model(model)
