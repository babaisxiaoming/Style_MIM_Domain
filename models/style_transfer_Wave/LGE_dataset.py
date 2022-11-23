import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image
import glob
import os

import albumentations as A

transformer = A.Compose([
    # 非破坏性转换
    A.OneOf([
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
        A.HorizontalFlip(),
        A.VerticalFlip(),
    ]),
    # 非刚体转换
    A.OneOf([
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
    ], p=0.8),
    A.OneOf([
        # A.GaussianBlur(blur_limit=3, always_apply=False, p=0.5),
        A.MedianBlur(blur_limit=5, always_apply=False, p=0.5)
    ]),
    A.RandomBrightnessContrast(p=0.8),
])


class LGEDataSet(data.Dataset):
    def __init__(self, list_path, max_iters=None, crop_size=224, pat_id=0, mode='fewshot'):
        self.list_path = list_path
        self.crop_size = crop_size
        search_path = os.path.join(list_path, "LGE_PNG/*_{}_lge*.png".format(pat_id))
        self.img_ids = glob.glob(search_path)
        self.label_ids = glob.glob(os.path.join(list_path, "LGE_LABEL_PNG/*_{}_lge*.png".format(pat_id)))
        if mode == 'fulldata':
            self.img_ids = glob.glob(os.path.join(list_path, "LGE_PNG/pat*lge*.png"))
            self.label_ids = glob.glob(os.path.join(list_path, "LGE_LABEL_PNG/pat*lge*.png"))
        if mode == 'partdata':
            self.img_ids = []
            self.label_ids = []
            for pat_id in range(6, 46):
                search_path = os.path.join(list_path, "LGE_PNG/*_{}_lge*.png".format(pat_id))
                imgs = glob.glob(search_path)
                labels = glob.glob(os.path.join(list_path, "LGE_LABEL_PNG/*_{}_lge*.png".format(pat_id)))
                for path in imgs:
                    self.img_ids.append(path)
                for path in labels:
                    self.label_ids.append(path)
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
        label = Image.open(datafiles["label"])
        img_w, img_h = image.size
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
        label_copy = label_copy[..., 0]  # H,W

        image = image[:, :, ::-1]
        image = image / 255.0
        image = image.transpose((2, 0, 1))

        # label_copy = label_copy[np.newaxis, :]
        # transformed = transformer(image=image, mask=label_copy)
        # image = transformed['image']
        # label_copy = transformed['mask']
        # label_copy = label_copy[0]

        return image.copy(), label_copy.copy(), name


if __name__ == '__main__':
    dst = LGEDataSet("/data02/imucs_data/machine58_img/wangxiaoming/dataset/MS_CMRSeg2019", mode='partdata')
    trainloader = data.DataLoader(dst, batch_size=1, shuffle=False)
    for i, data in enumerate(trainloader):
        imgs, label, _ = data
        print(imgs.shape)  # 1x3x224x224
        print(label.shape)
