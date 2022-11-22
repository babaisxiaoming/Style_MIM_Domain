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

import wandb
from pytorch_lightning.loggers import WandbLogger

from utils.metric.ms_cmrseg_dataset_metric import metrics
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.loss.jaccard_loss import jaccard_loss as iou_loss
from utils.loss.keep_largest_connected import keep_largest_connected_components_monai

# from models.convnext_reconv_up.conv_style_up import ConvNeXt_Up
# from models.convnext_up import ConvNeXt_Up
from models.uconvnext.UConvNeXt import UConvNeXt
from models.uconvnext.Conv_Rec import Rec_Decoder

from models.style_transfer.VGG import encoder, decoder, fc_encoder, fc_decoder
from models.style_transfer.style_transfer_model import style_transfer
from models.prior_pretrain_MAE.models_mae import mae_vit_large_patch16_dec512d8b as get_mae


class Interface(pl.LightningModule):
    def __init__(self, args, wandb_logger=None, visula_path=None):
        super().__init__()
        self.args = args
        self.wandb_logger = wandb_logger
        self.visula_path = visula_path

        # style_transfer_model
        self.vgg_encoder = encoder
        self.vgg_decoder = decoder
        self.style_encoder = fc_encoder
        self.style_decoder = fc_decoder
        self.load_style_model()

        # segment_model
        # self.seg_model = ConvNeXt_Up(in_channels=3, out_channels=4)
        self.seg_model = UConvNeXt()
        self.rec_model = Rec_Decoder()

        # loss
        self.ce_loss = nn.CrossEntropyLoss()
        self.loss_norm = nn.MSELoss()
        self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-6)

        # best_metric
        self.best_metric = 0

        self.save_hyperparameters(args)

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        s_data, s_label, s_name, t_data, t_label, t_name, s_data_freq = batch

        # 风格迁移
        s_data_temp = torch.clone(s_data).detach()
        t_data_temp = torch.clone(t_data).detach()
        times = 1 if self.current_epoch < self.args.warmup_epochs else 3
        for i in range(times):
            images_s_style, sampling = style_transfer(self.vgg_encoder,
                                                      self.vgg_decoder,
                                                      self.style_encoder,
                                                      self.style_decoder,
                                                      s_data_temp,
                                                      t_data_temp)
            images_s_style = torch.mean(images_s_style, dim=1)
            images_s_style = torch.stack([images_s_style, images_s_style, images_s_style], dim=1)

        # 分割模型
        s_pred, s_pred_norm, decoder_feats = self.seg_model(images_s_style)
        s_f_pred, s_f_pred_norm, _ = self.seg_model(s_data_freq)

        # 辅助重构
        seg_loss = self.rec_model(s_data, s_label, decoder_feats)

        # loss
        train_seg_loss = self.train_loss(s_pred, s_label.long()) + self.train_loss(s_f_pred, s_label.long())
        train_consistency_loss = self.loss_norm(s_f_pred, s_pred)
        train_rec_loss = seg_loss

        train_loss = train_seg_loss + 0.5 * train_rec_loss + 0.1 * train_consistency_loss

        self.log('train_seg_loss', train_seg_loss.item())
        self.log('train_consistency_loss', train_consistency_loss.item())
        self.log('train_rec_loss', train_rec_loss.item())
        self.log('loss', train_loss.item())

        return {'loss': train_loss, 'train_seg_loss': train_seg_loss, 'train_consistency_loss': train_consistency_loss,
                'train_rec_loss': train_rec_loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_train_seg_loss = torch.stack([x['train_seg_loss'] for x in outputs]).mean()
        avg_train_consistency_loss = torch.stack([x['train_consistency_loss'] for x in outputs]).mean()
        avg_train_rec_loss = torch.stack([x['train_rec_loss'] for x in outputs]).mean()

        self.log('avg_loss', avg_loss)
        self.log('avg_train_seg_loss', avg_train_seg_loss)
        self.log('avg_train_consistency_loss', avg_train_consistency_loss)
        self.log('avg_train_rec_loss', avg_train_rec_loss)

    def validation_step(self, batch, batch_idx):
        image, label = batch
        pred, pred_norm, _ = self.seg_model(image)

        # 恢复
        row = (256 - image.shape[2]) // 2
        line = (256 - image.shape[3]) // 2
        pred = F.pad(pred, (row, row, line, line), 'constant', 0)
        pred = F.interpolate(pred, (label.shape[1], label.shape[2]), mode='bicubic')

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

        # self.save_image_online(image, label, pred)

        # class_labels = {0: "bg", 1: "LV", 2: "MYO", 3: "RV"}
        # wandb.log(
        #     {"samples": wandb.Image(image, masks={
        #         "predictions": {
        #             "mask_data": pred[0],
        #             "class_labels": class_labels
        #         },
        #         "ground_truth": {
        #             "mask_data": label[0],
        #             "class_labels": class_labels
        #         }
        #     })})
        # wandb.Image(image, masks={
        #     "prediction": {"mask_data": pred[0], "class_labels": class_labels},
        #     "ground truth": {"mask_data": label[0], "class_labels": class_labels}})

        # self.log(
        #     wandb.Image(image, masks={
        #     "prediction": {"mask_data": pred[0], "class_labels": class_labels},
        #     "ground truth": {"mask_data": label[0], "class_labels": class_labels}}))

        # images = [img for img in image]
        # captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(label, pred)]
        # self.wandb_logger.log_image(key='sample_images', images=images, caption=captions)

        # wandb_logger = WandbLogger()
        # columns = ['image', 'ground truth', 'prediction']
        # data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(image, label, pred))]
        # wandb_logger.log_table(key='sample_table', columns=columns, data=data)

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
        pred, pred_norm, _ = self.seg_model(image, False)

        # 恢复
        row = (256 - image.shape[2]) // 2
        line = (256 - image.shape[3]) // 2
        pred = F.pad(pred, (row, row, line, line), 'constant', 0)
        pred = F.interpolate(pred, (label.shape[1], label.shape[2]), mode='bicubic')

        # 求最大联通域
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = keep_largest_connected_components_monai(pred)

        label = label.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        pred = np.array(pred).astype(np.uint16)

        # 这里可以继续替换 此时的输入是1x192x192(0-3)
        res = metrics(label, pred, apply_hd=False, apply_asd=False, modality='lge')

        val_dice_1 = torch.tensor(res['lv'][0])
        val_dice_2 = torch.tensor(res['rv'][0])
        val_dice_3 = torch.tensor(res['myo'][0])
        return {"val_dice_1": val_dice_1, "val_dice_2": val_dice_2, "val_dice_3": val_dice_3}

    def test_epoch_end(self, outputs):
        mean_val_dice_1 = torch.stack([x['val_dice_1'] for x in outputs]).mean()
        mean_val_dice_2 = torch.stack([x['val_dice_2'] for x in outputs]).mean()
        mean_val_dice_3 = torch.stack([x['val_dice_3'] for x in outputs]).mean()

        # 三个的平均
        mean_val_dice = torch.stack([mean_val_dice_1, mean_val_dice_2, mean_val_dice_3]).mean()

        if self.best_metric > mean_val_dice:
            self.best_metric = mean_val_dice
            wandb.run.summary["best_val_mean_dice"] = self.best_metric

        self.log('val_mean_dice_1', mean_val_dice_1)
        self.log('val_mean_dice_2', mean_val_dice_2)
        self.log('val_mean_dice_3', mean_val_dice_3)
        self.log('val_mean_dice', mean_val_dice)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD([
            {'params': self.seg_model.parameters(), 'lr': self.args.lr},
            {'params': self.rec_model.parameters(), 'lr': 1.5e-2},
        ], momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        # optimizer = torch.optim.AdamW(self.seg_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.args.warmup_epochs,
                                                  max_epochs=self.args.max_epoch)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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
        chkpt_dir = '/data02/GongZ_GRP/GzStuA/wxm/demo1/result/prior_mae/checkpoint-999.pth'
        checkpoint = torch.load(chkpt_dir)
        msg = self.prior_model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)

    def train_loss(self, pred, label):
        loss = self.ce_loss(pred, label)
        loss += iou_loss(true=label.long(), logits=pred)
        return loss

    def distill_loss(self, teacher_output, student_out):
        teacher_out = F.softmax(teacher_output, dim=-1)
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

    def save_image_online(self, image, label, pred):
        label = np.moveaxis(label, 0, -1)
        label = label.astype(np.uint8)
        label = cv2.resize(label, (224, 224))
        label = np.moveaxis(label, -1, 0)
        label = label[:, np.newaxis, :]

        pred = np.moveaxis(pred, 0, -1)
        pred = pred.astype(np.uint8)
        pred = cv2.resize(pred, (224, 224))
        pred = np.moveaxis(pred, -1, 0)
        pred = pred[:, np.newaxis, :]

        image = image.cpu().detach().numpy()

        # 原图+label
        blend_ori = [blend_images(image=img, label=lab, alpha=0.5, cmap="hsv", rescale_arrays=True) for img, lab in
                     list(zip(image, label))]
        # 原图+结果图
        blend_pred = [blend_images(image=img, label=lab, alpha=0.5, cmap="hsv", rescale_arrays=True) for img, lab in
                      list(zip(image, pred))]

        columns = ['image', 'ground truth', 'prediction']
        data = [[wandb.Image(np.moveaxis(x_i, 0, -1)), wandb.Image(np.moveaxis(y_i, 0, -1)),
                 wandb.Image(np.moveaxis(y_pred, 0, -1))] for x_i, y_i, y_pred in
                list(zip(image, blend_ori, blend_pred))]
        self.wandb_logger.log_table(key='sample_table', columns=columns, data=data)

        # images = [img for img in image]
        # captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(label, pred)]
        # self.wandb_logger.log_image(key='sample_images', images=images, caption=captions)

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

    def showblend(self, image, label, out, pred_mae, batch_idx=1):
        image = image.detach().cpu().numpy()[0]

        label = np.moveaxis(label, 0, -1)
        label = label.astype(np.uint8)
        label = cv2.resize(label, (224, 224))
        label = label[np.newaxis, :]

        out = np.moveaxis(out, 0, -1)
        out = out.astype(np.uint8)
        out = cv2.resize(out, (224, 224))
        out = out[np.newaxis, :]

        # 原图+label
        blend_ori = blend_images(image=image, label=label, alpha=0.5, cmap="hsv", rescale_arrays=True)
        # 原图+结果图
        blend_pred = blend_images(image=image, label=out, alpha=0.5, cmap="hsv", rescale_arrays=True)

        row = 2
        line = 3

        # HxWxC
        plt.figure("blend image and label")
        plt.subplot(row, line, 1)
        plt.imshow(np.moveaxis(image, 0, -1))
        plt.subplot(row, line, 2)
        plt.imshow(np.moveaxis(label, 0, -1))
        plt.subplot(row, line, 3)
        plt.imshow(np.moveaxis(blend_ori, 0, -1))

        plt.subplot(row, line, 4)
        plt.imshow(np.moveaxis(blend_pred, 0, -1))
        plt.subplot(row, line, 5)
        plt.imshow(np.moveaxis(pred_mae, 0, -1))

        save_path = self.visula_path
        assert os.path.exists(save_path)
        print(f'--- {batch_idx} saved in {save_path} ---')
        plt.savefig(save_path + f'/{batch_idx}.png')
