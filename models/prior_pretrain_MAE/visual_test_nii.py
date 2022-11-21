import os

import torch
import numpy as np

import matplotlib.pyplot as plt
from models.prior_pretrain_MAE import models_mae
from models.prior_pretrain_MAE.Prior_bSSFP_dataset import bSSFPDataSet

from tqdm import tqdm

from torch.utils.data import Dataset

dataset_train = bSSFPDataSet("/data02/imucs_data/machine58_img/wangxiaoming/dataset/MS_CMRSeg2019", crop_size=224)
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, num_workers=4, pin_memory=True)


def showblend(image_ori, label_ori, out_ori, im_paste_ori, save_path, idx):
    for i in range(0, image_ori.shape[0], 5):
        image = image_ori[i]
        label = label_ori[i]
        out = out_ori[i]
        im_paste = im_paste_ori[i]

        image = image.detach().cpu().numpy()
        image = image.astype(np.uint8)
        # image = np.moveaxis(image, -1, 0)

        # image = cv2.resize(image, (192, 192))
        # image = np.moveaxis(image, -1, 0)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = image[np.newaxis, :]
        # label = label[0, np.newaxis, :]
        # out = out[0, np.newaxis, :]

        label = label.detach().cpu().numpy()
        label = label.astype(np.uint8)
        # label = np.moveaxis(label, -1, 0)

        out = out.detach().cpu().numpy()

        '''
        重点
        '''
        # out = np.clip(out, 0, 3)  # 重点重点重点！！！！！！！！
        im_paste = im_paste.detach().cpu().numpy()

        row = 1
        line = 5

        im_paste_2 = np.around(im_paste)

        # HxWxC
        plt.figure("blend image and label")
        plt.subplot(row, line, 1)
        plt.imshow(image)
        plt.subplot(row, line, 2)
        plt.imshow(label)
        plt.subplot(row, line, 3)
        plt.imshow(out)
        plt.subplot(row, line, 4)
        plt.imshow(im_paste)
        plt.subplot(row, line, 5)
        plt.imshow(im_paste_2)
        # plt.subplot(row, line, 6)
        # plt.imshow(np.moveaxis(blend_3, 0, -1))

        plt.savefig(save_path + f'{idx}_{i}.png')


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 1
    image = torch.clip(image * 255, 0, 255).int()
    image = image.cpu().numpy()
    plt.imshow(image)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def run_one_image(img, model, save_path, idx):
    x = torch.tensor(img)

    x = x.cuda()
    model = model.cuda()

    # make it a batch-like
    # x = x.unsqueeze(dim=0)
    # x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y)

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 1)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach()

    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    showblend(x, im_masked, y, im_paste, save_path, idx)


os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# load model
chkpt_dir = '/data02/GongZ_GRP/GzStuA/wxm/demo1/result/prior_mae/checkpoint-999.pth'
model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
print('Model loaded.')
save_path = '/data02/GongZ_GRP/GzStuA/wxm/Style_MIM_Domain/result/Prior_PreTrain_MAE/'
for idx, i in tqdm(enumerate(data_loader_train)):
    mask = i
    # mask = mask.cuda()
    # model_mae = model_mae.cuda()
    #
    # _, pred, mask_ = model_mae(mask)
    # print(pred.shape)

    run_one_image(mask, model_mae, save_path, idx)
    # save_path = '/data02/WeiHX_GRP/WhxStuE/wxm/project/demo1/result/Prior_PreTrain_MAE/visual_test'
    # save_name = img_path.split('/')[-1]
    # save_path += f'/{save_name}'
    # print('MAE with pixel reconstruction:')
    # run_one_image(img, model_mae, save_path)
    #
    # print(f'save {idx}/{len(img_list)}')
