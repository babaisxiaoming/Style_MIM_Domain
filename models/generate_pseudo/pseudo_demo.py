import argparse
import os
import torch.nn.functional as F
import numpy as np

import torch
from torch.autograd import Variable
import tqdm
from torch.utils.data import DataLoader
from models.generate_pseudo import custom_transforms as tr
from torchvision import transforms

import torch.backends.cudnn as cudnn
import random

from datasets.MS_CMRSeg_PNG.bSSFP_dataset import bSSFPDataSet
from models.convnext_up import ConvNeXt_Up

import matplotlib.pyplot as plt


def showblend(image, label, prediction, prediction_pseudo, mask, batch_idx=1):
    prediction = torch.softmax(prediction, dim=1)
    prediction[prediction > 0.75] = 1.0  # 注意这里阈值的选择
    prediction[prediction <= 0.75] = 0.0
    mask = torch.argmax(mask, dim=1)
    prediction = torch.argmax(prediction, dim=1)
    prediction_pseudo = torch.argmax(prediction_pseudo, dim=1)

    image = image.detach().cpu().numpy()[0]
    label = label.detach().cpu().numpy()[0]
    prediction = prediction.detach().cpu().numpy()[0]
    prediction_pseudo = prediction_pseudo.detach().cpu().numpy()[0]
    mask = mask.detach().cpu().numpy()[0]

    label = label[np.newaxis, :]
    prediction = prediction[np.newaxis, :]
    prediction_pseudo = prediction_pseudo[np.newaxis, :]

    row = 1
    line = 5
    # HxWxC
    plt.figure("blend image and label")
    plt.subplot(row, line, 1)
    plt.imshow(np.moveaxis(image, 0, -1))
    plt.subplot(row, line, 2)
    plt.imshow(np.moveaxis(label, 0, -1))
    plt.subplot(row, line, 3)
    plt.imshow(np.moveaxis(prediction, 0, -1))
    plt.subplot(row, line, 4)
    plt.imshow(np.moveaxis(prediction_pseudo, 0, -1))
    plt.subplot(row, line, 5)
    plt.imshow(np.moveaxis(mask, 0, -1))

    save_path = '/data02/GongZ_GRP/GzStuA/wxm/Style_MIM_Domain/temp'
    assert os.path.exists(save_path)
    print(f'--- {batch_idx} saved in {save_path} ---')
    plt.savefig(save_path + f'/{batch_idx}.png')


bceloss = torch.nn.BCELoss(reduction='none')
seed = 3377
savefig = False
get_hd = True
model_save = True
if True:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--data-dir', default='/data02/imucs_data/machine58_img/wangxiaoming/dataset/Fundus')
    parser.add_argument('--out-stride', type=int, default=16)
    parser.add_argument('--sync-bn', type=bool, default=True)
    parser.add_argument('--freeze-bn', type=bool, default=False)
    parser.add_argument('--batchsize', type=int, default=8)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    # 1. dataset
    composed_transforms_train = transforms.Compose([
        tr.Resize(512),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    composed_transforms_test = transforms.Compose([
        tr.Resize(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    db_train = bSSFPDataSet("/data02/imucs_data/machine58_img/wangxiaoming/dataset/MS_CMRSeg2019", crop_size=224)
    train_loader = DataLoader(db_train, batch_size=args.batchsize, shuffle=False, num_workers=1)

    # 2. model
    model = ConvNeXt_Up(in_channels=3, out_channels=4)
    ckpt_path = '/data02/GongZ_GRP/GzStuA/wxm/Style_MIM_Domain/result/Cross_sa_vit_unetr_MS_CMRSeg'
    resume_checkpoint_path = os.path.join(ckpt_path, "epoch=890-val_mean_dice=0.744757.ckpt")
    ckpt = torch.load(resume_checkpoint_path)['state_dict']
    new_parms = {}
    for k, v in ckpt.items():
        if k.startswith('seg_model'):
            k = k[10:]
            if not k.startswith('MIM_model'):
                new_parms[k] = v
    mes = model.load_state_dict(new_parms)
    print(mes)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    if args.dataset == "Domain2":
        npfilename = '/data02/GongZ_GRP/GzStuA/wxm/Style_MIM_Domain/datasets/pseudo_MS_CMRSeg/pseudolabel_D2.npz'

    elif args.dataset == "Domain1":
        npfilename = './generate_pseudo/pseudolabel_D1.npz'

    npdata = np.load(npfilename, allow_pickle=True)
    pseudo_label_dic = npdata['arr_0'].item()
    uncertain_dic = npdata['arr_1'].item()
    proto_pseudo_dic = npdata['arr_2'].item()

    var_list = model.named_parameters()

    optim_gen = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.99))
    best_val_cup_dice = 0.0
    best_val_disc_dice = 0.0
    best_avg = 0.0

    iter_num = 0
    for epoch_num in tqdm.tqdm(range(2), ncols=70):
        with torch.no_grad():
            for batch_idx, (sample) in enumerate(train_loader):
                # data, target, img_name = sample['image'], sample['map'], sample['img_name']
                data, _, target, img_name = sample
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                # prediction, _, feature = model(data)
                prediction, feature = model(data)
                prediction_ori = prediction
                prediction = torch.softmax(prediction, dim=1)

                pseudo_label = [pseudo_label_dic.get(key) for key in img_name]
                uncertain_map = [uncertain_dic.get(key) for key in img_name]
                proto_pseudo = [proto_pseudo_dic.get(key) for key in img_name]

                pseudo_label = torch.from_numpy(np.asarray(pseudo_label)).float().cuda()
                uncertain_map = torch.from_numpy(np.asarray(uncertain_map)).float().cuda()
                proto_pseudo = torch.from_numpy(np.asarray(proto_pseudo)).float().cuda()

                # for param in model.parameters():
                #     param.requires_grad = True
                # optim_gen.zero_grad()

                target_0_obj = F.interpolate(pseudo_label[:, 0:1, ...], size=feature.size()[2:], mode='nearest')
                target_1_obj = F.interpolate(pseudo_label[:, 1:2, ...], size=feature.size()[2:], mode='nearest')
                target_2_obj = F.interpolate(pseudo_label[:, 2:3, ...], size=feature.size()[2:], mode='nearest')
                target_3_obj = F.interpolate(pseudo_label[:, 3:4, ...], size=feature.size()[2:], mode='nearest')
                target_0_bck = 1.0 - target_0_obj
                target_1_bck = 1.0 - target_1_obj
                target_2_bck = 1.0 - target_2_obj
                target_3_bck = 1.0 - target_3_obj

                mask_0_obj = torch.zeros(
                    [pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
                mask_0_bck = torch.zeros(
                    [pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
                mask_1_obj = torch.zeros(
                    [pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
                mask_1_bck = torch.zeros(
                    [pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
                mask_2_obj = torch.zeros(
                    [pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
                mask_2_bck = torch.zeros(
                    [pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
                mask_3_obj = torch.zeros(
                    [pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
                mask_3_bck = torch.zeros(
                    [pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
                mask_0_obj[uncertain_map[:, 0:1, ...] < 0.05] = 1.0
                mask_0_bck[uncertain_map[:, 0:1, ...] < 0.05] = 1.0
                mask_1_obj[uncertain_map[:, 1:2, ...] < 0.05] = 1.0
                mask_1_bck[uncertain_map[:, 1:2, ...] < 0.05] = 1.0
                mask_2_obj[uncertain_map[:, 2:3, ...] < 0.05] = 1.0
                mask_2_bck[uncertain_map[:, 2:3, ...] < 0.05] = 1.0
                mask_3_obj[uncertain_map[:, 3:4, ...] < 0.05] = 1.0
                mask_3_bck[uncertain_map[:, 3:4, ...] < 0.05] = 1.0
                mask = torch.cat(
                    (mask_0_obj * pseudo_label[:, 0:1, ...] + mask_0_bck * (1.0 - pseudo_label[:, 0:1, ...]),
                     mask_1_obj * pseudo_label[:, 1:2, ...] + mask_1_bck * (1.0 - pseudo_label[:, 1:2, ...]),
                     mask_2_obj * pseudo_label[:, 2:3, ...] + mask_2_bck * (1.0 - pseudo_label[:, 2:3, ...]),
                     mask_3_obj * pseudo_label[:, 3:4, ...] + mask_3_bck * (1.0 - pseudo_label[:, 3:4, ...])),
                    dim=1)

                mask_proto = torch.zeros([data.shape[0], 4, data.shape[2], data.shape[3]]).cuda()
                mask_proto[pseudo_label == proto_pseudo] = 1.0

                mask = mask * mask_proto

                # 8x2x512x512 8x2x512x512 float32
                # loss_seg_pixel = bceloss(prediction, pseudo_label)

                '''
                这里加可视化
                '''
                showblend(data, target, prediction_ori, pseudo_label, mask, batch_idx)

                # loss_seg = torch.sum(mask * loss_seg_pixel) / torch.sum(mask)
                # loss_seg.backward()
                # optim_gen.step()
                # iter_num = iter_num + 1

        if not os.path.exists('./logs/train_target'):
            os.mkdir('./logs/train_target')
        if args.dataset == 'Domain1':
            savefile = './logs/train_target/' + 'D1_' + 'checkpoint_%d.pth.tar' % epoch_num
        elif args.dataset == 'Domain2':
            savefile = './logs/train_target/' + 'D2_' + 'checkpoint_%d.pth.tar' % epoch_num
        if model_save:
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_mean_dice': best_avg,
                'best_cup_dice': best_val_cup_dice,
                'best_disc_dice': best_val_disc_dice,
            }, savefile)
