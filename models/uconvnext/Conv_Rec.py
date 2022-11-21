from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks import one_hot

from models.uconvnext import UConvNeXt
from models.mim.MIM_block_model import DeepInfoMax


# -------------------------- 分割深层监督 --------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Seg_Deep(nn.Module):
    def __init__(self,
                 in_channels,
                 scale,
                 ratio=16,
                 out_channels: int = 4):
        super().__init__()
        self.scale = scale
        self.channel_attention = ChannelAttention(in_planes=in_channels, ratio=ratio)
        self.spatial_attention = SpatialAttention()
        self.out = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.soft = nn.Sigmoid()

        self.ce_loss = nn.BCEWithLogitsLoss()

    def reparameterize(self, c_a, s_a, gate=1.0):
        std = s_a.mul(0.5).exp_()
        esp = torch.randn(*c_a.size()).cuda()
        # esp = torch.randn(*c_a.size())
        x = c_a + std * esp * gate
        return x

    def bottleneck(self, x, gate=1.0):
        c_a, s_a = self.channel_attention(x), self.spatial_attention(x)
        x = self.reparameterize(c_a, s_a, gate)
        return x

    def seg_loss(self, label, seg):
        if label.ndim == 3:
            label = label.unsqueeze(1)
            label = one_hot(label, num_classes=seg.size()[1], dim=1)

        if self.scale != 1:
            down_label = F.interpolate(label, scale_factor=1 / self.scale, mode='bilinear')
        else:
            down_label = label

        down_loss = self.ce_loss(seg, down_label)
        return down_loss

    def forward(self, label, x):
        x = self.bottleneck(x)
        x = self.soft(self.out(x))
        seg_loss = self.seg_loss(label, x)
        return x, seg_loss


# -------------------------- 重构深层监督 --------------------------
class Rec_Deep(nn.Module):
    def __init__(self,
                 scale,
                 in_dim=196,
                 out_dim=3,
                 in_dims=[128, 128, 64, 64, 32],
                 out_dims=[128, 64, 64, 32, 32],
                 layer=7,
                 kernel=3,
                 padding=1):
        super().__init__()
        self.scale = scale
        self.decoder = nn.Sequential()
        self.decoder.append(nn.Sequential(
            nn.Conv2d(in_dim, 128, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ))

        for i in range(layer - 2):
            block = nn.Sequential(
                nn.Conv2d(in_dims[i], out_dims[i], kernel_size=kernel, padding=padding),
                nn.BatchNorm2d(out_dims[i]),
                nn.ReLU(inplace=True))
            self.decoder.append(block)

        self.decoder.append(nn.Sequential(
            nn.Conv2d(out_dims[-1], out_dim, 1, 1),
            nn.Sigmoid()
        ))

        self.ce_loss = nn.BCEWithLogitsLoss()

    def rec_loss(self, image, rec):
        if self.scale != 1:
            down_image = F.interpolate(image, scale_factor=1 / self.scale, mode='bilinear')
        else:
            down_image = image

        down_loss = self.ce_loss(rec, down_image)
        return down_loss

    def forward(self, image, feat, pred):
        x = self.decoder(torch.cat((feat, pred), dim=1))
        rec_loss = self.rec_loss(image, x)
        return rec_loss


class Rec_Decoder(nn.Module):
    def __init__(self,
                 out_channels: int = 3,
                 feat_dims: Tuple[int, int, int, int] = [96, 192, 384, 768],
                 rec_dims: Tuple[int, int, int, int] = [384, 192, 96, 3],
                 decoder_depths: Tuple[int, int, int] = [3, 3, 3],
                 drop_path_rate: float = 0.1, ):
        super().__init__()

        self.seg_decoder_1 = Seg_Deep(in_channels=192, scale=8)
        self.seg_decoder_2 = Seg_Deep(in_channels=96, scale=4)
        self.seg_decoder_3 = Seg_Deep(in_channels=12, scale=1, ratio=1)

        self.rec_decoder_1 = Rec_Deep(in_dim=192 + 4, scale=8)
        self.rec_decoder_2 = Rec_Deep(in_dim=96 + 4, scale=4)
        self.rec_decoder_3 = Rec_Deep(in_dim=12 + 4, scale=1)

    def forward(self, image, label, decoder_feats):
        # 分割的深度监督
        s_1, s_2, s_3 = decoder_feats
        s_1, s_1_seg_loss = self.seg_decoder_1(label, s_1)
        s_2, s_2_seg_loss = self.seg_decoder_2(label, s_2)
        s_3, s_3_seg_loss = self.seg_decoder_3(label, s_3)
        seg_deep_loss = s_1_seg_loss + s_2_seg_loss + s_3_seg_loss
        '''
        torch.Size([1, 4, 28, 28])
        torch.Size([1, 4, 56, 56])
        torch.Size([1, 4, 224, 224])
        '''

        # 辅助任务深度监督
        r_1, r_2, r_3 = decoder_feats
        r_1_rec_loss = self.rec_decoder_1(image, r_1, s_1)
        r_2_rec_loss = self.rec_decoder_2(image, r_2, s_2)
        r_3_rec_loss = self.rec_decoder_3(image, r_3, s_3)
        rec_deep_loss = r_1_rec_loss + r_2_rec_loss + r_3_rec_loss

        return seg_deep_loss, rec_deep_loss
        # return seg_deep_loss


if __name__ == '__main__':
    image = torch.randn(1, 3, 224, 224).to(torch.float32)
    label = torch.randint(0, 1, (1, 224, 224)).to(torch.float32)

    # 主干网络 decoder 的feats
    m1 = torch.randn(1, 192, 28, 28)
    m2 = torch.randn(1, 96, 56, 56)
    m3 = torch.randn(1, 12, 224, 224)

    model = Rec_Decoder()
    y_hat = model(image, label, [m1, m2, m3])
    for i in y_hat:
        print(i)
