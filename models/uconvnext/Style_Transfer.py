from typing import Tuple

import torch
import torch.nn as nn


class Up_Basic(nn.Module):
    def __init__(self, dim, dim_scale=2):
        super().__init__()
        self.dim_scale = dim_scale
        self.expand = nn.ConvTranspose2d(dim, dim // dim_scale, kernel_size=2, stride=2, padding=0)
        self.conv = nn.Conv2d(dim // dim_scale, dim // dim_scale, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(dim // dim_scale)
        self.act = nn.LeakyReLU(True)

    def forward(self, x):
        x = self.expand(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class Up_4x(nn.Module):
    def __init__(self, dim, out_channels=4, dim_scale=4):
        super().__init__()
        self.up_basic = Up_Basic(dim)
        self.dim_scale = 2
        self.norm = nn.BatchNorm2d(dim // 8)
        self.act = nn.LeakyReLU(True)
        self.out = nn.Conv2d(dim // 8, out_channels, 1, 1)

    def forward(self, x):
        x = self.up_basic(x)  # 先2倍上采样
        B, C, H, W = x.size()
        x = x.view(B, -1, self.dim_scale * H, self.dim_scale * W)
        x = self.norm(x)
        x = self.act(x)
        x = self.out(x)
        return x


class Conv_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.ac = nn.ReLU(True)

    def forward(self, x):
        x = self.ac(self.bn(self.conv(x)))
        return x


class Up_Model(nn.Module):
    def __init__(self, dims=[768, 384, 192, 96], out_channels=3):
        super().__init__()

        self.upsample_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            upsample_layer = nn.Sequential(
                Conv_Block(dims[i]),
                Up_Basic(dims[i]),
            )
            self.upsample_layers.append(upsample_layer)
        self.upsample_layers.append(Up_4x(dims[-1], out_channels=out_channels))

    def forward(self, x):
        outs = []
        for idx, layer in enumerate(self.upsample_layers):
            x = layer(x)
            outs.append(x)
        return outs


class MLP_Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp_block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        x = self.mlp_block(x)
        return x


class Mean_Std(nn.Module):
    def __init__(self, channels=768):
        super().__init__()
        self.channels = channels
        self.in_mlp = MLP_Block(in_dim=channels * 2, out_dim=channels * 2)
        self.out_mlp = MLP_Block(in_dim=channels, out_dim=channels * 2)

    def mean_std(self, input, eps=1e-5, ndim=2):
        """
        Calculate channel-wise mean and standard deviation for the input features but reduce the dimensions
        Args:
            input: the latent feature of shape [B, C, H, W]
            eps: a small value to prevent calculation error of variance

        Returns:
        Channel-wise mean and standard deviation of the input features
        """
        B, C = input.size()[:2]
        feat_var = input.view(B, C, -1).var(dim=2) + eps
        if ndim == 2:
            feat_std = feat_var.sqrt().view(B, C)
            feat_mean = input.view(B, C, -1).mean(dim=2).view(B, C)
        elif ndim == 4:
            feat_std = feat_var.sqrt().view(B, C, 1, 1)
            feat_mean = input.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)

        return torch.cat([feat_mean, feat_std], dim=1)

    def adaptive_instance_normalization_with_noise(self, content_feat, style_feat):
        """
        Implementation of AdaIN of style transfer
        Args:
            content_feat: the content features of shape [B, C, H, W]
            style_feat: the style features of shape [B, C, H, W]

        Returns:
        The re-normalized features
        """
        size = content_feat.size()
        B, C = content_feat.size()[:2]
        style_mean = style_feat[:, :self.channels].view(B, C, 1, 1)
        style_std = style_feat[:, self.channels:].view(B, C, 1, 1)

        content_mean_std = self.mean_std(content_feat, ndim=4)
        content_mean, content_std = content_mean_std[:, :self.channels, :], content_mean_std[:, self.channels:, :]

        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def forward(self, source, target, alpha=1.0):
        mean_std = self.in_mlp(self.mean_std(target, ndim=2))
        mean, std = mean_std[:, :self.channels], mean_std[:, self.channels:]
        noise = torch.randn_like(mean)
        sampling = mean + noise * std
        recon_mean_std = self.out_mlp(sampling)

        feat = self.adaptive_instance_normalization_with_noise(source, recon_mean_std)
        feat = alpha * feat + (1 - alpha) * source

        return feat, mean, std, mean_std, recon_mean_std


class Style_Transfer(nn.Module):
    def __init__(self):
        super().__init__()

        self.mean_std_block = Mean_Std()
        self.up_block = Up_Model(dims=[768, 384, 192, 96], out_channels=3)
        self.transfer_loss = nn.MSELoss()

    def forward(self, source_feats, target_feats):
        feat, mean, std, mean_std, recon_mean_std = self.mean_std_block(source_feats[0], target_feats[0])

        # 这里本来是通过decoder恢复至224，然后再encoder提取特征进行；现在跳过这些步骤直接计算loss
        feats = self.up_block(feat)

        # loss
        loss_target = self.transfer_loss(feat, target_feats[0])
        for i in range(len(feats) - 1):
            loss_target += self.transfer_loss(feats[i], target_feats[i + 1])
        loss = self.transfer_loss(mean, std) + self.transfer_loss(recon_mean_std, mean_std) + loss_target
        return loss


if __name__ == '__main__':
    '''
    torch.Size([1, 96, 56, 56])
    torch.Size([1, 192, 28, 28])
    torch.Size([1, 384, 14, 14])
    torch.Size([1, 768, 7, 7])
    '''
    s1 = torch.randn(1, 96, 56, 56)
    s2 = torch.randn(1, 192, 28, 28)
    s3 = torch.randn(1, 384, 14, 14)
    s4 = torch.randn(1, 768, 7, 7)

    t1 = torch.randn(1, 96, 56, 56)
    t2 = torch.randn(1, 192, 28, 28)
    t3 = torch.randn(1, 384, 14, 14)
    t4 = torch.randn(1, 768, 7, 7)

    model = Style_Transfer()
    y_hat = model([s4, s3, s2, s1], [t4, t3, t2, t1])
    print(y_hat)