import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image
import glob
import os
import imgaug.augmenters as iaa
import albumentations as A
import cv2


class ImageProcessor:
    @staticmethod
    def simple_aug(image, mask):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.05), "y": (-0.1, 0.1)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-10, 10),  # rotate by -10 to +10 degrees
                    shear=(-12, 12),  # shear by -12 to +12 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode='constant'  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
            ],
            random_order=True
        )
        if image.ndim == 4:
            mask = np.array(mask)
            image_heavy, mask_heavy = seq(images=image, segmentation_maps=mask.astype(np.int32))
        else:
            image_heavy, mask_heavy = seq(images=image[np.newaxis, ...],
                                          segmentation_maps=mask[np.newaxis, ...].astype(np.int32))
            image_heavy, mask_heavy = image_heavy[0], mask_heavy[0]
        return image_heavy, mask_heavy


transformer = A.Compose([
    # 非破坏性转换
    A.OneOf([
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
        A.HorizontalFlip(),
        A.VerticalFlip(),
    ]),
    # 非刚体转换
    # A.OneOf([
    # A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    # A.GridDistortion(p=0.5),
    # A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
    # ], p=0.8),
    A.OneOf([
        A.GaussianBlur(blur_limit=3, always_apply=False, p=0.5),
        A.MedianBlur(blur_limit=5, always_apply=False, p=0.5)
    ]),
    # A.RandomBrightnessContrast(),
])


class bSSFPDataSet(data.Dataset):
    def __init__(self, list_path, max_iters=None, crop_size=224):
        self.list_path = list_path
        self.crop_size = crop_size
        self.img_ids = glob.glob(os.path.join(list_path, "BSSFP_PNG/*bSSFP*.png"))
        self.label_ids = glob.glob(os.path.join(list_path, "BSSFP_LABEL_PNG/*bSSFP*.png"))
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
            self.label_ids = self.label_ids * int(np.ceil(float(max_iters) / len(self.label_ids)))
        self.files = []
        self.id_to_trainid = {0: 0, 85: 1, 212: 2, 255: 3}
        for img_file, label_file in zip(self.img_ids, self.label_ids):
            name = os.path.basename(img_file)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')  # H,W,C
        img_w, img_h = image.size
        label = Image.open(datafiles["label"])
        if img_w != self.crop_size:
            border_size = int((img_w - self.crop_size) // 2)
            image = image.crop((border_size, border_size, img_w - border_size, img_h - border_size))
            label = label.crop((border_size, border_size, img_w - border_size, img_h - border_size))
        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        label_copy = np.asarray(label, np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label_copy == k] = v
        label_copy = np.expand_dims(label_copy, axis=-1)

        # image, label_copy = ImageProcessor.simple_aug(image=image, mask=label_copy)
        transformed = transformer(image=image, mask=label_copy)
        image = transformed['image']
        label_copy = transformed['mask']

        label_copy = label_copy[..., 0]  # H,W

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = image[:, :, ::-1]
        image = image[:, :, np.newaxis]
        image = image / 255.0
        image = image.transpose((2, 0, 1))  # C,H,W

        label_copy = label_copy[np.newaxis, :]
        return label_copy.copy()
        # return np.concatenate([image.copy(), label_copy.copy()], axis=0)


if __name__ == '__main__':
    dst = bSSFPDataSet("/data02/imucs_data/machine58_img/wangxiaoming/dataset/MS_CMRSeg2019", crop_size=224)
    trainloader = data.DataLoader(dst, batch_size=4, shuffle=False)
    for i, data in enumerate(trainloader):
        image = data
        print(image.shape)

        # image = image[0].numpy()
        # labels = labels[0].numpy()
        # plt.figure("blend image and label")
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.moveaxis(image, 0, -1))
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.moveaxis(labels, 0, -1))
        # plt.show()
        #
        # if i == 10:
        #     break
