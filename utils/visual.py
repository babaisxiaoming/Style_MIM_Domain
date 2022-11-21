import torch
from monai.utils import first, set_determinism
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from torch.utils.tensorboard import SummaryWriter
from monai.data import DataLoader, Dataset
from monai.visualize import blend_images, matshow3d, plot_2d_or_3d_image
import os
import glob
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

# from itkwidgets import view

'''
monai可视化
1. input_image 可视化
2. image blend label 可视化
tutorials:
https://github.com/Project-MONAI/tutorials/blob/main/modules/transform_visualization.ipynb
'''


def get_dataloader(data_dir):
    train_images = sorted(glob.glob(os.path.join(data_dir, "data", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "TCIA_pancreas_labels-02-05-2017", "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]

    transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="PLS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ])

    check_ds = Dataset(data=data_dicts, transform=transform)
    check_loader = DataLoader(check_ds, batch_size=1)
    data = first(check_loader)
    # print(f"image shape: {data['image'].shape}, label shape: {data['label'].shape}")
    return data


def show3d(data, figsize, every_n):
    matshow3d(volume=data['image'], fig=None, title='input image', figsize=figsize, every_n=every_n, frame_dim=-1,
              show=True, cmap="gray")


def showblend(image, label, figsize=(10, 10), every_n=20):
    blend = blend_images(image=image, label=label, alpha=0.5, cmap="hsv", rescale_arrays=True)
    # print(blend.shape)

    slips = blend.shape[-1]
    assert every_n < slips
    slice_index = [i for i in range(0, slips, every_n)]

    row = len(slice_index)
    line = 3

    plt.figure("blend image and label", figsize)
    for i in range(len(slice_index)):
        plt.subplot(row, line, (i * line + 1))
        plt.imshow(image[0, :, :, slice_index[i]], cmap='gray')
        plt.subplot(row, line, (i * line + 2))
        plt.imshow(label[0, :, :, slice_index[i]])
        plt.subplot(row, line, (i * line + 3))
        plt.imshow(torch.moveaxis(blend[:, :, :, slice_index[i]], 0, -1))

    plt.savefig('haha.jpg')


def show_tensorboard(path, data):
    if not os.path.exists(path):
        os.makedirs(path)
    plot_2d_or_3d_image(data=data, step=0, writer=SummaryWriter(log_dir=path), frame_dim=-1)


def show_image(data_dir):
    data = get_dataloader(data_dir)

    figsize = (10, 10)
    every_n = 20

    # 3d显示
    # show3d(data, figsize, every_n)

    # 混合显示
    showblend(data["image"][0], data["label"][0], figsize, every_n)

    # view(image=data["image"][0, 0, :, :, :] * 255, label_image=data["label"][0, 0, :, :, :] * 255, gradient_opacity=0.4)


data_dir = '/data02/imucs_data/machine58_img/wangxiaoming/dataset/Pancreas-CT/Pancreas-CT'
show_image(data_dir)
