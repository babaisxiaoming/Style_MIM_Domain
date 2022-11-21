import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks import one_hot

from models.mim.MIM_block_model import DeepInfoMax

from functools import partial
from monai.networks.blocks.dynunet_block import UnetOutBlock

from utils.loss.lovasz_losses import lovasz_softmax


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


class BalancedBCELoss(nn.Module):
    def __init__(self, target):
        super(BalancedBCELoss, self).__init__()
        self.eps = 1e-6
        weight = torch.tensor([torch.reciprocal(torch.sum(target == 0).float() + self.eps),
                               torch.reciprocal(torch.sum(target == 1).float() + self.eps),
                               torch.reciprocal(torch.sum(target == 2).float() + self.eps),
                               torch.reciprocal(torch.sum(target == 3).float() + self.eps)])
        self.criterion = nn.CrossEntropyLoss(weight)

    def forward(self, output, target):
        loss = self.criterion(output, target)

        return loss


class VAE_Decoder(nn.Module):
    def __init__(self,
                 in_dim=68,
                 in_dims=[128, 128, 64, 64, 32],
                 out_dims=[128, 64, 64, 32, 32],
                 layer=7,
                 kernel=3,
                 padding=1):
        super(VAE_Decoder, self).__init__()

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
            nn.Conv2d(out_dims[-1], 3, kernel_size=kernel, padding=padding),
            nn.Sigmoid()
        ))

    def forward(self, z, y):
        z = self.decoder(torch.cat((z, y), dim=1))
        return z


class Spatial_Attention(nn.Module):
    def __init__(self, in_channel):
        super(Spatial_Attention, self).__init__()
        self.activate = nn.Sequential(nn.Conv2d(in_channel, 1, kernel_size=1))

    def forward(self, x):
        actition = self.activate(x)
        out = torch.mul(x, actition)

        return out


class MIM(nn.Module):

    def __init__(self,
                 dims: [int, int, int] = [16, 32, 64],
                 out_channels: int = 4
                 ):
        super().__init__()

        # bottleneck
        # self.down2_fc1 = nn.Sequential(Spatial_Attention(dims[2]),
        #                                nn.InstanceNorm2d(dims[2]),
        #                                nn.Tanh())
        # self.down2_fc2 = nn.Sequential(nn.Conv2d(dims[2], dims[2], kernel_size=3, padding=1),
        #                                nn.InstanceNorm2d(dims[2]),
        #                                nn.Tanh())
        self.down2_fc1 = ChannelAttention(dims[2])
        self.down2_fc2 = SpatialAttention()

        # self.segdown2_seq = nn.Sequential(nn.Conv2d(dims[2], out_channels, kernel_size=3, padding=1))
        self.segdown2_seq = UnetOutBlock(spatial_dims=2, in_channels=dims[2], out_channels=out_channels)

        # self.down1_fc1 = nn.Sequential(Spatial_Attention(dims[1]),
        #                                nn.InstanceNorm2d(dims[1]),
        #                                nn.Tanh())
        # self.down1_fc2 = nn.Sequential(nn.Conv2d(dims[1], dims[1], kernel_size=3, padding=1),
        #                                nn.InstanceNorm2d(dims[1]),
        #                                nn.Tanh())
        self.down1_fc1 = ChannelAttention(dims[1])
        self.down1_fc2 = SpatialAttention()

        # self.segdown1_seq = nn.Sequential(nn.Conv2d(dims[1], out_channels, kernel_size=3, padding=1), )
        self.segdown1_seq = UnetOutBlock(spatial_dims=2, in_channels=dims[1], out_channels=out_channels)

        # self.fc1 = nn.Sequential(Spatial_Attention(dims[0]),
        #                          nn.InstanceNorm2d(dims[0]),
        #                          nn.Tanh())
        # self.fc2 = nn.Sequential(nn.Conv2d(dims[0], dims[0], kernel_size=3, padding=1),
        #                          nn.InstanceNorm2d(dims[0]),
        #                          nn.Tanh())

        self.fc1 = ChannelAttention(dims[0])
        self.fc2 = SpatialAttention()

        # self.segdown_seq = nn.Sequential(nn.Conv2d(dims[0], out_channels, kernel_size=3, padding=1))
        self.segdown_seq = UnetOutBlock(spatial_dims=2, in_channels=dims[0], out_channels=out_channels)

        self.soft = nn.Softmax2d()

        # seg_loss
        self.ce_loss = nn.CrossEntropyLoss()

        # reconstruct
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

        self.down2 = partial(F.interpolate, scale_factor=0.5, mode='area', recompute_scale_factor=True)
        self.down4 = partial(F.interpolate, scale_factor=0.25, mode='area', recompute_scale_factor=True)

        self.decoder_down4 = VAE_Decoder(in_dim=64 + 4)
        self.decoder_down2 = VAE_Decoder(in_dim=32 + 4)
        self.decoder_down = VAE_Decoder(in_dim=16 + 4)

        # MIM_loss
        self.MIM_block = DeepInfoMax(0.5, 1.0, 0.1)

    def reparameterize(self, mu, logvar, gate):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp * gate
        return z

    def bottleneck_0(self, h, gate):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar, gate)
        return z, mu, logvar

    def bottleneck_1(self, h, gate):
        mu, logvar = self.down1_fc1(h), self.down1_fc2(h)
        z = self.reparameterize(mu, logvar, gate)
        return z, mu, logvar

    def bottleneck_2(self, h, gate):
        mu, logvar = self.down2_fc1(h), self.down2_fc2(h)
        z = self.reparameterize(mu, logvar, gate)
        return z, mu, logvar

    def forward_bottleneck(self, deout2, deout3, deout4, gate=1.0):
        feat_down4, down4_mu, down4_logvar = self.bottleneck_2(deout2, gate)
        segout_down4 = self.segdown2_seq(feat_down4)

        feat_down2, down2_mu, down2_logvar = self.bottleneck_1(deout3, gate)
        segout_down2 = self.segdown1_seq(feat_down2)

        feat_down, down_mu, down_logvar = self.bottleneck_0(deout4, gate)
        segout_down = self.segdown_seq(feat_down)

        return feat_down4, down4_mu, down4_logvar, segout_down4, feat_down2, down2_mu, down2_logvar, segout_down2, feat_down, down_mu, down_logvar, segout_down

    def forward_seg_loss(self, segout_down, segout_down2, segout_down4, s_label, s_label_down2, s_label_down4):
        seg_loss_down = lovasz_softmax(F.softmax(segout_down, dim=1), s_label)
        seg_loss_down2 = lovasz_softmax(F.softmax(segout_down2, dim=1), s_label_down2)
        seg_loss_down4 = lovasz_softmax(F.softmax(segout_down4, dim=1), s_label_down4)

        return 0.5 * seg_loss_down + 0.3 * seg_loss_down2 + 0.2 * seg_loss_down4

    def forward_reconstruct(self,
                            feat_down, down_mu, down_logvar, segout_down,
                            feat_down2, down2_mu, down2_logvar, segout_down2,
                            feat_down4, down4_mu, down4_logvar, segout_down4,
                            s_data, s_label, t_data,
                            s_data_down2, s_label_down2, t_data_down2,
                            s_data_down4, s_label_down4, t_data_down4):
        recon_down = self.decoder_down(feat_down, s_label)
        BCE_down = self.bce_loss(recon_down, s_data)
        KLD_down = -0.5 * torch.mean(1 + down_logvar - down_mu.pow(2) - down_logvar.exp())

        recon_down2 = self.decoder_down2(feat_down2, s_label_down2)
        BCE_down2 = self.bce_loss(recon_down2, s_data_down2)
        KLD_down2 = -0.5 * torch.mean(1 + down2_logvar - down2_mu.pow(2) - down2_logvar.exp())

        recon_down4 = self.decoder_down4(feat_down4, s_label_down4)
        BCE_down4 = self.bce_loss(recon_down4, s_data_down4)
        KLD_down4 = -0.5 * torch.mean(1 + down4_logvar - down4_mu.pow(2) - down4_logvar.exp())

        '''
        kldlamda : 1.0
        '''
        kldlamda = 1.0
        balanced_loss = 0.5 * (BCE_down + torch.mul(KLD_down, kldlamda)) + \
                        0.3 * (BCE_down2 + torch.mul(KLD_down2, kldlamda)) + \
                        0.2 * (BCE_down4 + torch.mul(KLD_down4, kldlamda))
        return recon_down, recon_down2, recon_down4, balanced_loss

    def forward(self, deout2, deout3, deout4, gate,
                s_data, s_label, t_data,
                s_data_down2, s_label_down2, t_data_down2,
                s_data_down4, s_label_down4, t_data_down4):
        # bottleneck
        feat_down4, down4_mu, down4_logvar, segout_down4, \
        feat_down2, down2_mu, down2_logvar, segout_down2, \
        feat_down, down_mu, down_logvar, segout_down = self.forward_bottleneck(deout2, deout3, deout4, gate)

        s_label = torch.cat([s_label, s_label], dim=0)
        s_label_down2 = torch.cat([s_label_down2, s_label_down2], dim=0)
        s_label_down4 = torch.cat([s_label_down4, s_label_down4], dim=0)
        # s_label = one_hot(torch.cat([s_label, s_label], dim=0).unsqueeze(1), num_classes=4)
        # s_label_down2 = one_hot(torch.cat([s_label_down2, s_label_down2], dim=0).unsqueeze(1), num_classes=4)
        # s_label_down4 = one_hot(torch.cat([s_label_down4, s_label_down4], dim=0).unsqueeze(1), num_classes=4)

        # s_label = s_label.unsqueeze(1)
        # s_label_down2 = s_label_down2.unsqueeze(1)
        # s_label_down4 = s_label_down4.unsqueeze(1)
        # seg_loss
        seg_loss = self.forward_seg_loss(segout_down, segout_down2, segout_down4, s_label, s_label_down2, s_label_down4)

        s_data_down2 = self.down2(s_data_down2)
        s_data_down4 = self.down4(s_data_down4)
        s_label = one_hot(s_label.unsqueeze(1), num_classes=4)
        s_label_down2 = one_hot(s_label_down2.unsqueeze(1), num_classes=4)
        s_label_down4 = one_hot(s_label_down4.unsqueeze(1), num_classes=4)
        # reconstruct
        recon_down, recon_down2, recon_down4, balanced_loss = \
            self.forward_reconstruct(
                feat_down, down_mu, down_logvar, segout_down,
                feat_down2, down2_mu, down2_logvar, segout_down2,
                feat_down4, down4_mu, down4_logvar, segout_down4,
                s_data, s_label, t_data,
                s_data_down2, s_label_down2, t_data_down2,
                s_data_down4, s_label_down4, t_data_down4)

        # MIM_loss
        loss_mi = self.MIM_block(segout_down4, segout_down2, segout_down, recon_down4, recon_down2, recon_down)

        return seg_loss + balanced_loss + loss_mi
        # return seg_loss


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    dout2 = torch.randn(1, 64, 56, 56).cuda()
    dout3 = torch.randn(1, 32, 112, 112).cuda()
    dout4 = torch.randn(1, 16, 224, 224).cuda()

    s_data = torch.randn(1, 3, 224, 224).cuda()
    s_label = torch.randint(0, 4, (1, 224, 224)).cuda()
    t_data = torch.randn(1, 3, 224, 224).cuda()

    s_data_down2 = torch.randn(1, 3, 112, 112).cuda()
    s_label_down2 = torch.randint(0, 4, (1, 112, 112)).cuda()
    t_data_down2 = torch.randn(1, 3, 112, 112).cuda()

    s_data_down4 = torch.randn(1, 3, 56, 56).cuda()
    s_label_down4 = torch.randint(0, 4, (1, 56, 56)).cuda()
    t_data_down4 = torch.randn(1, 3, 56, 56).cuda()

    model = MIM().cuda()
    y_hat = model(dout2, dout3, dout4, 1.0,
                  s_data, s_label, t_data,
                  s_data_down2, s_label_down2, t_data_down2,
                  s_data_down4, s_label_down4, t_data_down4)

    print(y_hat)
