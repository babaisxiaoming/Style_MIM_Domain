import torch
from torch.utils import data
from datasets.MS_CMRSeg_PNG.LGE_dataset import LGEDataSet
from datasets.MS_CMRSeg_PNG.bSSFP_dataset import bSSFPDataSet
from skimage import transform
import numpy as np
import random

from pylab import plt


# 理想低通滤波器
def ILPF(image, d0, n):
    H = np.empty_like(image, dtype=float)
    _, M, N = image.shape
    mid_x = int(M / 2)
    mid_y = int(N / 2)
    for y in range(0, M):
        for x in range(0, N):
            d = np.sqrt((x - mid_x) ** 2 + (y - mid_y) ** 2)
            if d <= d0:
                H[:, y, x] = 1 ** n
            else:
                H[:, y, x] = 0 ** n
    return H


# 高斯低通滤波器
def GLPF(image, d0, n):
    H = np.empty_like(image, float)
    _, M, N = image.shape
    mid_x = M / 2
    mid_y = N / 2
    for x in range(0, M):
        for y in range(0, N):
            d = np.sqrt((x - mid_x) ** 2 + (y - mid_y) ** 2)
            H[:, x, y] = np.exp(-d ** n / (2 * d0 ** n))
    return H


# 巴特沃斯低通滤波器
def BLPF(image, d0, n):
    H = np.empty_like(image, float)
    _, M, N = image.shape
    mid_x = int(M / 2)
    mid_y = int(N / 2)
    for y in range(0, M):
        for x in range(0, N):
            d = np.sqrt((x - mid_x) ** 2 + (y - mid_y) ** 2)
            H[:, y, x] = 1 / (1 + (d / d0) ** (n))
    return H


# 傅立叶变换
def fourier_transform(image, input=False, output=False):
    if input:
        fftImg = np.fft.fft2(image, axes=(-2, -1))  # 傅里叶变换
        # fftImgShift = np.abs(fftImg)  # 傅里叶变换后坐标移动到图像中心
        fftImgShift = np.fft.fftshift(fftImg)  # 傅里叶变换后坐标移动到图像中心
    if output:
        fftImgShift = np.fft.ifftshift(image)  # 逆傅立叶变换
        fftImgShift = np.fft.ifft2(fftImgShift)
        fftImgShift = np.real(fftImgShift)  # 傅里叶反变换后取频域

    return fftImgShift


def low_freq_mutate_np(s_amp_np, t_amp_np, L=0.1):
    a_src = np.fft.fftshift(s_amp_np, axes=(-2, -1))
    a_trg = np.fft.fftshift(t_amp_np, axes=(-2, -1))

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


def paper_change(source, target, L=0.1):
    s_fft = np.fft.fft2(source, axes=(-2, -1))
    s_amp_np, s_pha_np = np.abs(s_fft), np.angle(s_fft)

    t_fft = np.fft.fft2(target, axes=(-2, -1))
    t_amp_np, t_pha_np = np.abs(t_fft), np.angle(t_fft)

    amp_src_ = low_freq_mutate_np(s_amp_np, t_amp_np, L=L)

    s_fft_src_ = amp_src_ * np.exp(1j * s_pha_np)
    src_in_trg = np.fft.ifft2(s_fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)

    t_fft_src_ = amp_src_ * np.exp(1j * t_pha_np)
    trg_in_src = np.fft.ifft2(t_fft_src_, axes=(-2, -1))
    trg_in_src = np.real(trg_in_src)

    # 再次mixup
    # _, h, w = src_in_trg.shape
    # b = (np.floor(np.amin((h, w)) * L)).astype(int)
    # c_h = np.floor(h / 2.0).astype(int)
    # c_w = np.floor(w / 2.0).astype(int)
    # print (b)
    # h1 = c_h - b
    # h2 = c_h + b + 1
    # w1 = c_w - b
    # w2 = c_w + b + 1
    # ratio = random.randint(1, 10) / 10
    # src_in_trg[:, h1:h2, w1:w2] = src_in_trg[:, h1:h2, w1:w2] * ratio + trg_in_src[:, h1:h2, w1:w2] * (1 - ratio)

    # 标准mixup
    ratio = 0.1 * np.random.random(1)
    src_in_trg = src_in_trg * (1 - ratio) + trg_in_src * ratio

    return src_in_trg


def mix(image_1, image_2, L=0.1):
    image_1_f = fourier_transform(image_1, input=True)
    image_1_t = image_1_f * ILPF(image_1, 60, 2)

    image_2_f = fourier_transform(image_2, input=True)
    image_2_t = image_2_f * ILPF(image_2, 60, 2)

    _, h, w = image_1_t.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)
    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    ratio = random.randint(1, 10) / 10

    image_1_t[h1:h2, w1:w2] = image_1_t[h1:h2, w1:w2] * ratio + image_2_t[h1:h2, w1:w2] * (1 - ratio)
    # image_1_t = image_1_t * ratio + image_2_t * (1 - ratio)

    mixImage = fourier_transform(image_1_t, output=True)
    # mixImage = normal(mixImage, output=True)
    return mixImage


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
        # img_freq = mix(s_data, t_data)
        s_data = np.array(s_data).astype(np.float32)
        img_freq = paper_change(s_data, t_data)
        img_freq = np.clip(img_freq, 0, 255).astype(np.float32)

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
        # print(s_data.shape)
        # print(s_label.shape)
        # print(s_name)
        # print(t_data.shape)
        # print(t_label.shape)
        # print(t_name)
        # print(img_freq.shape)
        save_image(s_data, s_label, t_data, t_label, img_freq)
        break
