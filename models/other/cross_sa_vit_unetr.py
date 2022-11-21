from typing import Tuple, Union
import math

import torch
import torch.nn as nn

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from models.Vit_UNETR.vit import vit_base_patch16_224


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
              "The distribution of values may be incorrect.", )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()
        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class vit_up(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: Tuple[int, int],
            feature_size: int = 16,
            hidden_size: int = 768,
            mlp_dim: int = 3072,
            num_heads: int = 12,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = False,
            res_block: bool = True,
            dropout_rate: float = 0.0,
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
        )
        self.hidden_size = hidden_size

        self.vit = vit_base_patch16_224(pretrained=False, in_chans=in_channels)

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=2, in_channels=feature_size, out_channels=out_channels)  # type: ignore

        self.up = UnetrPrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights["state_dict"]:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.position_embeddings_3d"]
            )
            self.vit.patch_embedding.cls_token.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.cls_token"]
            )
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.weight"]
            )
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.bias"]
            )

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights["state_dict"]["module.transformer.norm.weight"])
            self.vit.norm.bias.copy_(weights["state_dict"]["module.transformer.norm.bias"])

    def forward_encoder(self, x):
        x = self.vit.patch_embedding(x)  # 输出 1x2744x768(2744=14*14*14)

        hidden_states_out = []
        for blk in self.vit.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.vit.norm(x)

        return x, hidden_states_out

    def forward(self, x_in_1, x_in_2, train):
        x, x_2, x_fusion, hidden, hidden_2, hidden_fusion = self.vit(x_in_1, x_in_2, train)

        if train:
            enc1 = self.encoder1(x_in_1)
            x2, x3, x4 = hidden[3], hidden[6], hidden[9]
            enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
            enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
            enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
            dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
            dec3 = self.decoder5(dec4, enc4)
            dec2 = self.decoder4(dec3, enc3)
            dec1 = self.decoder3(dec2, enc2)
            out = self.decoder2(dec1, enc1)

        enc1_2 = self.encoder1(x_in_2)
        x2_2, x3_2, x4_2 = hidden_2[3], hidden_2[6], hidden_2[9]
        enc2_2 = self.encoder2(self.proj_feat(x2_2, self.hidden_size, self.feat_size))
        enc3_2 = self.encoder3(self.proj_feat(x3_2, self.hidden_size, self.feat_size))
        enc4_2 = self.encoder4(self.proj_feat(x4_2, self.hidden_size, self.feat_size))
        dec4_2 = self.proj_feat(x_2, self.hidden_size, self.feat_size)
        dec3_2 = self.decoder5(dec4_2, enc4_2)
        dec2_2 = self.decoder4(dec3_2, enc3_2)
        dec1_2 = self.decoder3(dec2_2, enc2_2)
        out_2 = self.decoder2(dec1_2, enc1_2)

        if train:
            x2_fusion, x3_fusion, x4_fusion = hidden_fusion[3], hidden_fusion[6], hidden_fusion[9]
            enc2_fusion = self.encoder2(self.proj_feat(x2_fusion, self.hidden_size, self.feat_size))
            enc3_fusion = self.encoder3(self.proj_feat(x3_fusion, self.hidden_size, self.feat_size))
            enc4_fusion = self.encoder4(self.proj_feat(x4_fusion, self.hidden_size, self.feat_size))
            dec4_fusion = self.proj_feat(x_fusion, self.hidden_size, self.feat_size)
            dec3_fusion = self.decoder5(dec4_fusion, enc4_fusion)
            dec2_fusion = self.decoder4(dec3_fusion, enc3_fusion)
            dec1_fusion = self.decoder3(dec2_fusion, enc2_fusion)
            # out_fusion = self.decoder2(dec1_fusion, enc1_fusion)  # 1x32x112x112  # enc1:1x16x224x224
            out_fusion = self.up(dec1_fusion)

        logits = None
        x_2 = self.proj_feat(x_2, self.hidden_size, self.feat_size)
        logits_2 = self.out(out_2)
        logits_fusion = None

        if train:
            x = self.proj_feat(x, self.hidden_size, self.feat_size)

            x_fusion = self.proj_feat(x_fusion, self.hidden_size, self.feat_size)
            logits = self.out(out)
            logits_fusion = self.out(out_fusion)

        return x, x_2, x_fusion, logits, logits_2, logits_fusion


if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    x_2 = torch.rand(1, 3, 224, 224)
    model = vit_up(in_channels=3, out_channels=4, img_size=(224, 224))
    train = True
    y_hat = model(x, x_2, train)
    for i in y_hat:
        print(i.shape)
