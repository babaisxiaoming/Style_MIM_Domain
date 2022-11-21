import torch
from torch.utils import data
from datasets.MS_CMRSeg_PNG.LGE_dataset import LGEDataSet
from datasets.MS_CMRSeg_PNG.bSSFP_dataset import bSSFPDataSet
from skimage import transform
import numpy as np
import random

from pylab import plt


def extract_amp_spectrum(img_np):
    # trg_img is of dimention CxHxW (C = 3 for RGB image and 1 for slice)

    fft = np.fft.fft2(img_np, axes=(-2, -1))
    amp_np, pha_np = np.abs(fft), np.angle(fft)

    return amp_np


def low_freq_mutate_np(amp_src, amp_trg, L=0.1):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_trg, axes=(-2, -1))

    _, h, w = a_src.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)
    # print (b)
    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    ratio = random.randint(1, 10) / 10

    a_src[:, h1:h2, w1:w2] = a_src[:, h1:h2, w1:w2] * ratio + a_trg[:, h1:h2, w1:w2] * (1 - ratio)
    a_src = np.fft.ifftshift(a_src, axes=(-2, -1))
    return a_src


def source_to_target_freq(src_img, amp_trg, L=0.1):
    # exchange magnitude
    # input: src_img, trg_img
    # src_img = src_img.transpose((2, 0, 1))
    src_img_np = src_img  # .cpu().numpy()
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np(amp_src, amp_trg, L=L)

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    # get the mutated image
    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    # return src_in_trg.transpose(1, 2, 0)
    return src_in_trg


class TrainSet(data.Dataset):
    def __init__(self, data_path, len_data=1000):
        self.len_data = len_data
        self.source_data = bSSFPDataSet(data_path, crop_size=224)
        self.target_data = LGEDataSet(data_path, crop_size=224, mode='partdata')
        self.l_source = self.source_data.__len__()
        self.l_target = self.target_data.__len__()

        # target_pseudo
        # target_pseudo_path = '/data02/GongZ_GRP/GzStuA/wxm/Style_MIM_Domain/datasets/pseudo_MS_CMRSeg/target_LGE_pseudo.npz'
        # self.target_pseudo = np.load(target_pseudo_path, allow_pickle=True)
        # self.pseudo_label_dic = self.target_pseudo['arr_0'].item()
        # self.mask_dic = self.target_pseudo['arr_1'].item()

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        s_data, s_label, s_name = self.source_data.__getitem__(index % self.l_source)
        t_data, t_label, t_name = self.target_data.__getitem__(index % self.l_target)

        '''
        傅立叶变换
        '''
        img = np.array(s_data).astype(np.float32)
        # amp_trg = extract_amp_spectrum(t_data.transpose(2, 0, 1))
        amp_trg = extract_amp_spectrum(t_data)
        img_freq = source_to_target_freq(img, amp_trg, L=0.1)
        img_freq = np.clip(img_freq, 0, 255).astype(np.float32)

        '''
        正常的可视化与增广后的可视化
        '''
        # t_pseudo_label = self.pseudo_label_dic.get(t_name.split('.')[0])
        # t_mask = self.mask_dic.get(t_name.split('.')[0])
        # t_mask = np.argmax(t_mask, axis=0)


        '''
        获取下采样的图像，分别下采样 x2,x4
        '''
        # H, W = s_data.shape[1], s_data.shape[2]
        # s_data_down2 = transform.resize(s_data, (3, H // 2, W // 2), order=3, mode='edge', preserve_range=True)
        # s_label_down2 = transform.resize(s_label, (H // 2, W // 2), order=3, mode='edge', preserve_range=True)
        # t_data_down2 = transform.resize(t_data, (3, H // 2, W // 2), order=3, mode='edge', preserve_range=True)
        #
        # s_data_down4 = transform.resize(s_data, (3, H // 4, W // 4), order=3, mode='edge', preserve_range=True)
        # s_label_down4 = transform.resize(s_label, (H // 4, W // 4), order=3, mode='edge', preserve_range=True)
        # t_data_down4 = transform.resize(t_data, (3, H // 4, W // 4), order=3, mode='edge', preserve_range=True)

        # return s_data, s_label, s_name, t_data, t_name, \
        #        s_data_down2, s_label_down2, t_data_down2, \
        #        s_data_down4, s_label_down4, t_data_down4

        return s_data, s_label, s_name, t_data, t_label, t_name, img_freq


def save_image(s_data, s_label, t_data, t_label, img_freq):
    s_data = s_data.numpy()[1]
    s_label = s_label.numpy()[1]
    t_data = t_data.numpy()[1]
    t_label = t_label.numpy()[1]
    img_freq = img_freq.numpy()[1]

    row = 3
    line = 2
    plt.figure()
    plt.subplot(row, line, 1)
    plt.imshow(np.moveaxis(s_data, 0, -1))
    plt.subplot(row, line, 2)
    plt.imshow(s_label[:, :, np.newaxis])
    plt.subplot(row, line, 3)
    plt.imshow(np.moveaxis(t_data, 0, -1))
    plt.subplot(row, line, 4)
    plt.imshow(np.moveaxis(t_label, 0, -1))
    plt.subplot(row, line, 5)
    plt.imshow(np.moveaxis(img_freq, 0, -1))
    plt.show()


if __name__ == '__main__':
    dst = TrainSet("/data02/imucs_data/machine58_img/wangxiaoming/dataset/MS_CMRSeg2019")
    trainloader = data.DataLoader(dst, batch_size=4, shuffle=False)
    for i, data in enumerate(trainloader):
        s_data, s_label, s_name, t_data, t_label, t_name, img_freq = data
        print(s_data.shape)
        print(s_label.shape)
        print(s_name)
        print(t_data.shape)
        print(t_label.shape)
        print(t_name)
        # print(t_pseudo_label.shape)
        # print(t_mask.shape)
        print(img_freq.shape)
        # save_image(s_data, s_label, t_data, t_label, img_freq)
        break

    '''
    torch.Size([4, 3, 224, 224])
    torch.Size([4, 224, 224])
    torch.Size([4, 3, 224, 224])
    torch.Size([4, 3, 112, 112])
    torch.Size([4, 112, 112])
    torch.Size([4, 3, 112, 112])
    torch.Size([4, 3, 56, 56])
    torch.Size([4, 56, 56])
    torch.Size([4, 3, 56, 56])
    '''
