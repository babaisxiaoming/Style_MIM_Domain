import glob
import os

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import models_mae


def showblend(image, label, out, save_path):
    image = image.detach().cpu().numpy()
    image = np.clip(image, 0, 1)
    image = image * 255.
    image = image.astype(np.uint8)
    image = np.moveaxis(image, -1, 0)
    # image = cv2.resize(image, (192, 192))
    # image = np.moveaxis(image, -1, 0)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = image[np.newaxis, :]
    # label = label[0, np.newaxis, :]
    # out = out[0, np.newaxis, :]

    label = label.detach().cpu().numpy()
    label = np.clip(label, 0, 1)
    label = label * 255.
    label = label.astype(np.uint8)
    label = np.moveaxis(label, -1, 0)

    out = out.detach().cpu().numpy()
    out = np.clip(out, 0, 1)
    out = out * 255.
    out = out.astype(np.uint8)
    out = np.moveaxis(out, -1, 0)

    # out->one_hot
    # out = one_hot(torch.from_numpy(out), num_classes=4, dim=0)
    # out = out.numpy()

    # 原图+label
    # blend = blend_images(image=image, label=label, alpha=0.5, cmap="hsv", rescale_arrays=True)
    # 原图+结果图
    # blend_1 = blend_images(image=image, label=out[1:2, :, :], alpha=0.5, cmap="hsv", rescale_arrays=True)
    # blend_2 = blend_images(image=image, label=out[2:3, :, :], alpha=0.5, cmap="hsv", rescale_arrays=True)
    # blend_3 = blend_images(image=image, label=out[3:4, :, :], alpha=0.5, cmap="hsv", rescale_arrays=True)

    row = 1
    line = 3

    # HxWxC
    plt.figure("blend image and label")
    plt.subplot(row, line, 1)
    plt.imshow(np.moveaxis(image, 0, -1))
    plt.subplot(row, line, 2)
    plt.imshow(np.moveaxis(label, 0, -1))
    plt.subplot(row, line, 3)
    plt.imshow(np.moveaxis(out, 0, -1))

    # plt.subplot(row, line, 4)
    # plt.imshow(np.moveaxis(blend_1, 0, -1))
    # plt.subplot(row, line, 5)
    # plt.imshow(np.moveaxis(blend_2, 0, -1))
    # plt.subplot(row, line, 6)
    # plt.imshow(np.moveaxis(blend_3, 0, -1))

    plt.savefig(save_path)


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


def run_one_image(img, model, save_path):
    x = torch.tensor(img)

    x = x.cuda()
    model = model.cuda()

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

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

    # # make the plt figure larger
    # plt.rcParams['figure.figsize'] = [24, 24]
    #
    # plt.subplot(1, 4, 1)
    # # x[0]: 224x224x1 torch.float64
    # show_image(x[0], "original")
    #
    # plt.subplot(1, 4, 2)
    # # im_masked[0]: 224x224x1 torch.float64
    # show_image(im_masked[0], "masked")
    #
    # plt.subplot(1, 4, 3)
    # # y[0]: 224x224x1 torch.float64
    # show_image(y[0], "reconstruction")
    #
    # plt.subplot(1, 4, 4)
    # show_image(im_paste[0], "reconstruction + visible")
    #
    # plt.show()
    # plt.savefig(save_path)
    showblend(x[0], im_masked[0], y[0], save_path)


os.environ['CUDA_VISIBLE_DEVICES'] = '5'

root_dir = '/data02/imucs_data/machine58_img/wangxiaoming/dataset/MS_CMRSeg2019/LGE_LABEL_PNG/'
img_list = glob.glob(os.path.join(root_dir, '*.png'))

# load model
chkpt_dir = '/data02/WeiHX_GRP/WhxStuE/wxm/project/demo1/result/Prior_PreTrain_MAE/checkpoint-999.pth'
model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
print('Model loaded.')

for idx, i in enumerate(img_list):
    # load an image
    # img_path = '/data02/imucs_data/machine58_img/wangxiaoming/dataset/MS_CMRSeg2019/LGE_LABEL_PNG/pat_44_lge_5.png'
    img_path = os.path.join(root_dir, i)
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.
    img = img[:, :, np.newaxis]

    assert img.shape == (224, 224, 1)

    save_path = '/data02/WeiHX_GRP/WhxStuE/wxm/project/demo1/result/Prior_PreTrain_MAE/visual_test'
    save_name = img_path.split('/')[-1]
    save_path += f'/{save_name}'
    print('MAE with pixel reconstruction:')
    run_one_image(img, model_mae, save_path)

    print(f'save {idx}/{len(img_list)}')
