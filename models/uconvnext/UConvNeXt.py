from typing import Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath


# ######################## ConvNeXt Block ########################

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


# ######################## ConvNeXt Encoder Block ########################

class Encoder(nn.Module):
    def __init__(self, in_chans=3, dims=[96, 192, 384, 768], depths=[3, 3, 9, 3],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        # downsample
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # block
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.out_indices = out_indices

        # layer_norm
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)


# ######################## ConvNeXt Decoder Block ########################

# class Up_Basic(nn.Module):
#     def __init__(self, dim, dim_scale=2):
#         super().__init__()
#         self.dim_scale = dim_scale
#         self.expand = nn.Conv2d(dim, dim_scale * dim, 1, 1)
#         self.norm = LayerNorm(dim // dim_scale, eps=1e-6, data_format="channels_first")
#
#     def forward(self, x):
#         _, C, _, _ = x.size()
#         x = self.expand(x)
#         x = rearrange(x, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=self.dim_scale, p2=self.dim_scale,
#                       c=C // self.dim_scale).contiguous()
#         x = self.norm(x)
#         return x

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
    def __init__(self, dim, num_class=4, dim_scale=4):
        super().__init__()
        self.up_basic = Up_Basic(dim)
        self.dim_scale = 2
        self.norm = nn.BatchNorm2d(dim // 8)
        self.act = nn.LeakyReLU(True)
        self.out = nn.Conv2d(dim // 8, num_class, 1, 1)

    def forward(self, x):
        x = self.up_basic(x)  # 先2倍上采样
        B, C, H, W = x.size()
        x = x.view(B, -1, self.dim_scale * H, self.dim_scale * W)
        x = self.norm(x)
        x = self.act(x)
        x = self.out(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_class=4, dims=[768, 384, 192, 96], depths=[3, 3, 3],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2]):
        super().__init__()

        # upsample
        self.upsample_layers = nn.ModuleList()
        for i in range(3):
            upsample_layer = nn.Sequential(
                Up_Basic(dims[i]),
            )
            self.upsample_layers.append(upsample_layer)
        self.upsample_layers.append(Up_4x(dims[-1], num_class=num_class))

        # concat
        dims = dims[1:]
        self.convat_block = nn.ModuleList()
        for i in range(3):
            layer = nn.Sequential(
                nn.Conv2d(2 * dims[i], dims[i], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(dims[i]),
                nn.LeakyReLU(True)
            )
            self.convat_block.append(layer)

        # block
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(3):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        # layer_norm
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(3):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward(self, downsample):
        outs = []
        up_layer = None
        for idx, layer in enumerate(downsample):
            if idx == 0:
                up_layer = self.upsample_layers[idx](layer)
            else:
                up_layer = torch.cat([up_layer, layer], dim=1)
                up_layer = self.convat_block[idx - 1](up_layer)
                up_layer = self.stages[idx - 1](up_layer)
                if idx in self.out_indices:
                    norm_layer = getattr(self, f'norm{idx - 1}')
                    up_layer = norm_layer(up_layer)
                up_layer = self.upsample_layers[idx](up_layer)
            outs.append(up_layer)
        return tuple(outs)


class UConvNeXt(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 4,
            dims: Tuple[int, int, int, int] = [96, 192, 384, 768],
            encoder_depths: Tuple[int, int, int, int] = [3, 3, 9, 3],
            decoder_depths: Tuple[int, int, int] = [3, 3, 3],
            drop_path_rate: float = 0.1,

    ) -> None:
        super(UConvNeXt, self).__init__()

        self.encoder = Encoder(in_chans=in_channels, dims=dims, depths=encoder_depths, drop_path_rate=drop_path_rate)

        self.decoder = Decoder(num_class=out_channels, dims=tuple(reversed(dims)), depths=decoder_depths,
                               drop_path_rate=drop_path_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        '''
        torch.Size([1, 96, 56, 56])
        torch.Size([1, 192, 28, 28])
        torch.Size([1, 384, 14, 14])
        torch.Size([1, 768, 7, 7])
        '''
        x1, x2, x3, x4 = self.encoder(x)

        out = self.decoder([x4, x3, x2, x1])

        return out[-1], x4, [x4, x3, x2, x1]


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = UConvNeXt()
    y_hat = model(x)
    for i in y_hat:
        print(i.shape)


    def summary_model(model):
        torchinfo.summary(model, (3, 224, 224), batch_dim=0,
                          col_names=('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'), verbose=1)


    summary_model(model)
