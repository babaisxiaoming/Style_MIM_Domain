from torch.utils.data import Dataset
import glob
import numpy as np
import SimpleITK as sitk
import torch
import albumentations as A

import os
import cv2
import nibabel as nib

transformer = A.Compose([
    A.Resize(height=224, width=224)
])


def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


def read_img(pat_id, img_len, file_path):
    images = []

    for im in range(img_len):
        img = cv2.imread(os.path.join(file_path, "pat_{}_{}_{}.png".format(pat_id, 'lge', im)))
        images.append(img)
    return np.array(images)


def crop_volume(vol, crop_size=112):
    return np.array(vol[:,
                    int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                    int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size])


class LGE_SET(Dataset):
    def __init__(self, dir):
        self.imgs = np.zeros((1, 3, 224, 224))
        self.labs = np.zeros((1, 192, 192))
        # self.labs = np.zeros((1, 224, 224))

        # train:6:46
        # test:1:6
        for pat_id in range(6, 46):
            # label 192x192
            mask_path = os.path.join(dir, 'LGE_ALL/patient{}_{}_manual.nii.gz'.format(pat_id, 'LGE'))
            nimg, affine, header = load_nii(mask_path)
            nimg = nimg.T
            nimg = np.where(nimg == 200, 1, nimg)
            nimg = np.where(nimg == 500, 2, nimg)
            nimg = np.where(nimg == 600, 3, nimg)

            # image 3x224x224x224
            vol_resize = read_img(pat_id, nimg.shape[0], file_path=os.path.join(dir, 'LGE_PNG'))
            vol_resize = crop_volume(vol_resize, crop_size=112)
            x_batch = np.array(vol_resize, np.float32) / 255.
            x_batch = np.moveaxis(x_batch, -1, 1)

            self.imgs = np.concatenate((self.imgs, x_batch), axis=0)

            # resize为224x224
            # nimg = np.moveaxis(nimg, 0, -1)
            # nimg = cv2.resize(nimg, (224, 224))
            # nimg = np.moveaxis(nimg, -1, 0)
            self.labs = np.concatenate((self.labs, nimg), axis=0)

        self.imgs = self.imgs[1:, :, :, :]
        self.labs = self.labs[1:, :, :]

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, item):
        npimg = self.imgs[item]
        nplab = self.labs[item]

        return torch.from_numpy(npimg).type(dtype=torch.FloatTensor), \
               torch.from_numpy(nplab).type(dtype=torch.LongTensor)


if __name__ == '__main__':
    from torch.utils.data import Dataset, DataLoader

    path = "/data02/imucs_data/machine58_img/wangxiaoming/dataset/MS_CMRSeg2019"

    SourceData = LGE_SET(path)
    SourceData_loader = DataLoader(SourceData, batch_size=12, shuffle=True, num_workers=2, pin_memory=True,
                                   drop_last=True)
    for i in SourceData_loader:
        img, label = i
        print(img.shape)  # 12x1x224x224
        print(label.shape)  # 12x224x224
