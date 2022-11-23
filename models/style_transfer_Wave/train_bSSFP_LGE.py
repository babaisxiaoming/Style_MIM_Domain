from torch.utils import data
from LGE_dataset import LGEDataSet
from bSSFP_dataset import bSSFPDataSet
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

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        s_data, s_label, s_name = self.source_data.__getitem__(index % self.l_source)
        t_data, t_label, t_name = self.target_data.__getitem__(index % self.l_target)

        '''
        傅立叶变换
        '''
        # img = np.array(s_data).astype(np.float32)
        # amp_trg = extract_amp_spectrum(t_data)
        # img_freq = source_to_target_freq(img, amp_trg, L=0.1)
        # img_freq = np.clip(img_freq, 0, 255).astype(np.float32)

        # return s_data, s_label, s_name, t_data, t_label, t_name, img_freq
        return s_data, s_label, s_name, t_data, t_label, t_name


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
        s_data, s_label, s_name, t_data, t_label, t_name = data
        print(s_data.shape)
        print(s_label.shape)
        print(s_name)
        print(t_data.shape)
        print(t_label.shape)
        print(t_name)
        # save_image(s_data, s_label, t_data, t_label, img_freq)
        break
