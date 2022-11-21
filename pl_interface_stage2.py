import pytorch_lightning as pl
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2

from monai.losses import DiceLoss, DiceCELoss
from monai.visualize import blend_images

import matplotlib.pyplot as plt

from utils.metric.ms_cmrseg_dataset_metric import metrics
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.loss.jaccard_loss import jaccard_loss as iou_loss
from utils.loss.keep_largest_connected import keep_largest_connected_components_monai

from models.convnext_up import ConvNeXt_Up
# from models.other.convnext_uper_model.ConvNeXt_UperNet import ConvNeXt_Uper

from models.style_transfer.VGG import encoder, decoder, fc_encoder, fc_decoder
from models.style_transfer.style_transfer_model import style_transfer
from models.prior_pretrain_MAE.models_mae import mae_vit_large_patch16_dec512d8b as get_mae


class Interface(pl.LightningModule):
    def __init__(self, args, visula_path=None):
        super().__init__()
        self.args = args
        self.visula_path = visula_path

        # segment_model
        self.seg_model = ConvNeXt_Up(in_channels=3, out_channels=4)
        # 快速验证
        ckpt_path = '/data02/GongZ_GRP/GzStuA/wxm/Style_MIM_Domain/result/Cross_sa_vit_unetr_MS_CMRSeg'
        resume_checkpoint_path = os.path.join(ckpt_path, "epoch=890-val_mean_dice=0.744757.ckpt")
        ckpt = torch.load(resume_checkpoint_path)['state_dict']
        new_parms = {}
        for k, v in ckpt.items():
            if k.startswith('seg_model'):
                k = k[10:]
                if not k.startswith('MIM_model'):
                    new_parms[k] = v
        self.seg_model.load_state_dict(new_parms)

        # loss
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        # self.bceloss = torch.nn.BCELoss(reduction='none')
        # self.loss_norm = nn.MSELoss()
        self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-6)

        self.save_hyperparameters(args)

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        s_data, s_label, s_name, t_data, t_label, t_name, t_pseudo_label, t_mask = batch

        pred, pred_norm = self.seg_model(t_data, True)

        # loss
        # loss_dice = self.dice_loss(pred, torch.argmax(t_pseudo_label, dim=1).unsqueeze(1))
        loss_seg_pixel = self.ce_loss(torch.softmax(pred, dim=1), t_pseudo_label)
        # loss_seg_pixel = loss_dice + 0.5 * loss_ce
        pseudo_loss = torch.sum(t_mask * loss_seg_pixel) / torch.sum(t_mask)

        # 可视化
        # self.showblend(t_data, t_label, images_s_style, pred_prior, masked, pred_mae_round, batch_idx=batch_idx)

        train_loss = pseudo_loss

        self.log('loss', train_loss.item())
        return {'loss': train_loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        self.log('avg_loss', avg_loss)

    def validation_step(self, batch, batch_idx):
        image, label = batch
        pred, pred_norm = self.seg_model(image, False)

        # 恢复
        row = (256 - image.shape[2]) // 2
        line = (256 - image.shape[3]) // 2
        pred = F.pad(pred, (row, row, line, line), 'constant', 0)
        pred = F.interpolate(pred, (label.shape[1], label.shape[2]), mode='bicubic')

        val_loss = self.dice_loss(pred, label.unsqueeze(1))

        # 求最大联通域
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = keep_largest_connected_components_monai(pred, num_class=self.args.num_class)

        label = label.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        pred = np.array(pred).astype(np.uint16)

        res = metrics(label, pred, apply_hd=False, apply_asd=False, modality='lge')
        val_dice_1 = torch.tensor(res['lv'][0])
        val_dice_2 = torch.tensor(res['rv'][0])
        val_dice_3 = torch.tensor(res['myo'][0])
        return {"val_dice_1": val_dice_1, "val_dice_2": val_dice_2, "val_dice_3": val_dice_3}

    def validation_epoch_end(self, outputs):
        mean_val_dice_1 = torch.stack([x['val_dice_1'] for x in outputs]).mean()
        mean_val_dice_2 = torch.stack([x['val_dice_2'] for x in outputs]).mean()
        mean_val_dice_3 = torch.stack([x['val_dice_3'] for x in outputs]).mean()

        # 三个的平均
        mean_val_dice = torch.stack([mean_val_dice_1, mean_val_dice_2, mean_val_dice_3]).mean()

        self.log('val_mean_dice_1', mean_val_dice_1)
        self.log('val_mean_dice_2', mean_val_dice_2)
        self.log('val_mean_dice_3', mean_val_dice_3)
        self.log('val_mean_dice', mean_val_dice)

    def test_step(self, batch, batch_idx):
        image, label = batch
        pred, pred_norm = self.seg_model(image, False)

        pred_prior = torch.argmax(pred, dim=1)

        # 这个不能要
        # pred_prior = keep_largest_connected_components_monai(pred_prior, num_class=self.args.num_class)  # 新加了最大联通域
        _, pred_mae, mask = self.prior_model(pred_prior.unsqueeze(1), mask_ratio=0.75)
        pred_mae = self.prior_model.unpatchify(pred_mae)
        pred_mae = torch.einsum('nchw->nhwc', pred_mae)

        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.prior_model.patch_embed.patch_size[0] ** 2 * 1)  # (N, H*W, p*p*3)
        mask = self.prior_model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach()

        self.showblend(image, label, pred_prior, mask, pred_mae, batch_idx=batch_idx)

    def test_epoch_end(self, outputs):
        pass
        # mean_val_dice_1 = torch.stack([x['val_dice_1'] for x in outputs]).mean()
        # mean_val_dice_2 = torch.stack([x['val_dice_2'] for x in outputs]).mean()
        # mean_val_dice_3 = torch.stack([x['val_dice_3'] for x in outputs]).mean()
        #
        # # 三个的平均
        # mean_val_dice = torch.stack([mean_val_dice_1, mean_val_dice_2, mean_val_dice_3]).mean()
        #
        # self.log('val_mean_dice_1', mean_val_dice_1)
        # self.log('val_mean_dice_2', mean_val_dice_2)
        # self.log('val_mean_dice_3', mean_val_dice_3)
        # self.log('val_mean_dice', mean_val_dice)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.seg_model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        # optimizer = torch.optim.AdamW(self.seg_model.parameters(), lr=self.args.lr, weight_decay=1e-5)
        # optimizer = torch.optim.Adam(self.seg_model.parameters(), lr=self.args.lr, betas=(0.9, 0.99))
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.args.warmup_epochs,
                                                  max_epochs=self.args.max_epoch)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        # return optimizer

    def load_style_model(self):
        self.vgg_encoder.eval()
        self.style_encoder.eval()
        self.vgg_decoder.eval()
        self.style_decoder.eval()

        self.vgg_encoder.load_state_dict(torch.load(self.args.vgg_encoder))
        self.vgg_encoder = nn.Sequential(*list(self.vgg_encoder.children())[:31])
        try:
            self.vgg_decoder.load_state_dict(torch.load(self.args.vgg_decoder))
        except:
            self.vgg_decoder.load_state_dict(torch.load(self.args.vgg_decoder)['model_state_dict'])
            print("decoder load from state dict")
        try:
            self.style_encoder.load_state_dict(torch.load(self.args.style_encoder))
        except:
            self.style_encoder.load_state_dict(torch.load(self.args.style_encoder)['model_state_dict'])
            print("fc_decoder load from state dict")
        try:
            self.style_decoder.load_state_dict(torch.load(self.args.style_decoder))
        except:
            self.style_decoder.load_state_dict(torch.load(self.args.style_decoder)['model_state_dict'])
            print("fc_encoder load from state dict")

    def load_prior_model(self):
        self.prior_model.eval()
        chkpt_dir = '/data02/GongZ_GRP/GzStuA/wxm/Style_MIM_Domain/result/Prior_PreTrain_MAE/label_argument.pth'
        checkpoint = torch.load(chkpt_dir)
        msg = self.prior_model.load_state_dict(checkpoint['model'])
        print(msg)

    def train_loss(self, pred, label):
        loss = self.ce_loss(pred, label)
        loss += iou_loss(true=label.long(), logits=pred)
        return loss

    def distill_loss(self, teacher_output, student_out):
        teacher_out = F.softmax(teacher_output, dim=1)
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        return loss.mean()

    def kl_loss(self, student_out, teacher_output):
        logp_x = F.log_softmax(student_out.unsqueeze(1), dim=1)
        p_y = F.softmax(teacher_output.unsqueeze(1), dim=1)
        kl_loss = F.kl_div(logp_x, p_y)
        return kl_loss

    def consistency_loss(self, pred_norm):
        norm_loss = 0
        loss_norm = nn.MSELoss()
        for norm_id in range(pred_norm.size()[0] // 2):
            norm_loss += (pred_norm[norm_id] - pred_norm[norm_id + pred_norm.size()[0] // 2]).norm(p=2, dim=(1, 2))
        pred_norm = norm_loss / (pred_norm.size()[0] // 2)

        loss = loss_norm(pred_norm, torch.zeros(pred_norm.size()).float().cuda())
        return loss

    def save_image(self, image, image_type, batch_idx, slice):
        output = image.detach().cpu().numpy()[0]
        output = np.clip(output, 0, 1)
        output = output * 255.
        output = output.astype(np.uint8)
        output = np.moveaxis(output, 0, -1)

        save_path = os.path.join(self.visula_path, 'style_image')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        image_name = f'{image_type}_{batch_idx}_{slice}.jpg'
        cv2.imwrite(os.path.join(save_path, image_name), output)

    def showblend(self, image, label, images_s_style, out, mask, pred_mae, batch_idx=1):
        images_s_style = images_s_style.to(image.dtype)

        image = image.detach().cpu().numpy()[0]
        label = label.detach().cpu().numpy()[0]
        images_s_style = images_s_style.detach().cpu().numpy()[0]
        out = out.detach().cpu().numpy()[0]
        mask = mask.detach().cpu().numpy()[0]
        pred_mae = pred_mae.detach().cpu().numpy()[0]

        images_s_style = np.clip(images_s_style, 0, 1)

        label = label.astype(np.uint8)
        label = label[np.newaxis, :]

        out = out.astype(np.uint8)
        # out = out[np.newaxis, :]

        # 原图+label
        blend_ori = blend_images(image=image, label=label, alpha=0.5, cmap="hsv", rescale_arrays=True)
        # 原图+结果图
        blend_pred = blend_images(image=image, label=out, alpha=0.5, cmap="hsv", rescale_arrays=True)

        # pred_mae = np.clip(pred_mae, 0, 3)
        out = np.moveaxis(out, 0, -1)
        pred_mae = out * (1 - mask) + pred_mae * mask
        out = np.moveaxis(out, -1, 0)

        row = 2
        line = 4

        # HxWxC
        plt.figure("blend image and label")
        plt.subplot(row, line, 1)
        plt.imshow(np.moveaxis(image, 0, -1))
        plt.subplot(row, line, 2)
        plt.imshow(np.moveaxis(label, 0, -1))
        plt.subplot(row, line, 3)
        plt.imshow(np.moveaxis(images_s_style, 0, -1))

        plt.subplot(row, line, 4)
        plt.imshow(np.moveaxis(blend_ori, 0, -1))
        plt.subplot(row, line, 5)
        plt.imshow(np.moveaxis(blend_pred, 0, -1))

        plt.subplot(row, line, 6)
        plt.imshow(np.moveaxis(out, 0, -1))
        plt.subplot(row, line, 7)
        plt.imshow(mask)
        plt.subplot(row, line, 8)
        plt.imshow(pred_mae)

        save_path = self.visula_path
        assert os.path.exists(save_path)
        print(f'--- {batch_idx} saved in {save_path} ---')
        plt.savefig(save_path + f'/{batch_idx}.png')
