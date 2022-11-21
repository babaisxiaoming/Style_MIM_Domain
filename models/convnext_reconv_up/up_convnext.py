from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial
from timm.models.layers import trunc_normal_, DropPath


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


class Up_Basic(nn.Module):
    def __init__(self, in_size, in_dim, dim_scale=2):
        super().__init__()
        self.in_size = in_size
        self.in_dim = in_dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(in_dim, 2 * in_dim, bias=False)
        self.norm = nn.LayerNorm(in_dim // dim_scale)

    def forward(self, x):
        H, W = self.in_size
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // (self.dim_scale ** 2))
        x = self.norm(x)
        x = rearrange(x, 'b h w c-> b c h w', h=H * 2, w=W * 2)
        return x


class Out_Block(nn.Module):
    def __init__(self, in_size, in_dim, out_dim, dim_scale=4):
        super().__init__()
        self.in_size = in_size
        self.in_dim = in_dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(in_dim, 16 * in_dim, bias=False)
        self.norm = nn.LayerNorm(in_dim)

        self.out_conv = nn.Conv2d(in_dim, out_dim, 1, 1)

    def forward(self, x):
        H, W = self.in_size
        x = rearrange(x, 'b c h w-> b (h w) c').contiguous()
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=4, p2=4, c=C // (self.dim_scale ** 2)).contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c-> b c h w', h=H * 4, w=W * 4).contiguous()

        x = self.out_conv(x)
        return x


class Reverse_ConvNeXt(nn.Module):
    def __init__(self, basic_size=7, depths=[3, 9, 3, 3], dims=[1024, 512, 256, 128], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 4]):
        super().__init__()

        self.up_layers = nn.ModuleList()
        for i in range(3):
            up_layer = nn.Sequential(
                Up_Basic((basic_size * 2 ** i, basic_size * 2 ** i), in_dim=dims[i])
            )
            self.up_layers.append(up_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(1, 4):
            stage = nn.Sequential(
                nn.Conv2d(dims[i] * 2, dims[i], 1, 1),
                # *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                #         layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(1, 4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.out = Out_Block(in_size=(56, 56), in_dim=128, out_dim=4)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        outs = []
        for i in range(3):
            hidden_state = self.up_layers[i](x[i])
            '''
            torch.Size([1, 512, 14, 14])
            torch.Size([1, 256, 28, 28])
            torch.Size([1, 128, 56, 56])
            torch.Size([1, 64, 112, 112])
            '''
            hidden_state = torch.cat([x[i + 1], hidden_state], dim=1)
            hidden_state = self.stages[i](hidden_state)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i + 1}')
                x_out = norm_layer(hidden_state)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)

        out = self.out(x[-1])
        return out


if __name__ == '__main__':
    x1 = torch.randn(1, 128, 56, 56)
    x2 = torch.randn(1, 256, 28, 28)
    x3 = torch.randn(1, 512, 14, 14)
    x4 = torch.randn(1, 1024, 7, 7)

    hidden_state = [x4, x3, x2, x1]
    model = Reverse_ConvNeXt(depths=[3, 9, 3, 3], dims=[1024, 512, 256, 128])
    y_hat = model(hidden_state)
    print(y_hat.shape)
    # x = torch.randn(1, 1024, 14, 14)
    # model = Reverse_ConvNeXt(depths=[3, 9, 3, 3], dims=[1024, 512, 256, 128])
    # y_hat = model(x)
