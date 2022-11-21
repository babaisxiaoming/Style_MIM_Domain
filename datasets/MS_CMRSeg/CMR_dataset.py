from torch.utils.data import Dataset
import glob
import numpy as np
import SimpleITK as sitk
import torch
import albumentations as A

transformer = A.Compose([
    A.RandomCrop(height=160, width=160, always_apply=True),
    A.Resize(height=224, width=224)
])


class LGE_TrainSet(Dataset):
    def __init__(self, dir, sample_num, train):
        self.imgdir = dir + '/LGE/'

        if train:
            self.imgsname = glob.glob(self.imgdir + 'LGE_Test/*LGE.nii*')
        else:
            self.imgsname = glob.glob(self.imgdir + 'LGE_Vali/*LGE.nii*')

        imgs = np.zeros((1, 192, 192))
        labs = np.zeros((1, 192, 192))
        for img_num in range(sample_num):
            itkimg = sitk.ReadImage(self.imgsname[img_num])
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1

            imgs = np.concatenate((imgs, npimg), axis=0)

            labname = self.imgsname[img_num].replace('.nii', '_manual.nii')
            itklab = sitk.ReadImage(labname)
            nplab = sitk.GetArrayFromImage(itklab)
            nplab = (nplab == 200) * 1 + (nplab == 500) * 2 + (nplab == 600) * 3

            labs = np.concatenate((labs, nplab), axis=0)

        self.imgs = imgs[1:, :, :]
        self.labs = labs[1:, :, :]
        self.imgs.astype(np.float32)
        self.labs.astype(np.float32)

    def __getitem__(self, item):
        npimg = self.imgs[item]
        nplab = self.labs[item]

        transformed = transformer(image=npimg, mask=nplab)
        npimg = transformed['image']
        nplab = transformed['mask']

        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor), \
               torch.from_numpy(nplab).type(dtype=torch.LongTensor)

    def __len__(self):
        return self.imgs.shape[0]


class C0_TrainSet(Dataset):
    def __init__(self, dir, sample_num):
        self.imgdir = dir + '/C0/'

        self.imgsname = glob.glob(self.imgdir + '*C0.nii*')

        imgs = np.zeros((1, 192, 192))
        labs = np.zeros((1, 192, 192))
        for img_num in range(sample_num):
            itkimg = sitk.ReadImage(self.imgsname[img_num])
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,220*240*1

            imgs = np.concatenate((imgs, npimg), axis=0)

            labname = self.imgsname[img_num].replace('.nii', '_manual.nii')
            itklab = sitk.ReadImage(labname)
            nplab = sitk.GetArrayFromImage(itklab)
            nplab = (nplab == 200) * 1 + (nplab == 500) * 2 + (nplab == 600) * 3

            labs = np.concatenate((labs, nplab), axis=0)

        self.imgs = imgs[1:, :, :]
        self.labs = labs[1:, :, :]
        self.imgs.astype(np.float32)
        self.labs.astype(np.float32)

    def __getitem__(self, item):
        npimg = self.imgs[item]
        nplab = self.labs[item]

        transformed = transformer(image=npimg, mask=nplab)
        npimg = transformed['image']
        nplab = transformed['mask']

        return torch.from_numpy(npimg).unsqueeze(0).type(dtype=torch.FloatTensor), \
               torch.from_numpy(nplab).type(dtype=torch.LongTensor)

    def __len__(self):
        return self.imgs.shape[0]


class Train_Dataset(Dataset):
    def __init__(self, dir, train):
        # dir = '/data02/imucs_data/machine58_img/wangxiaoming/dataset/MMWHS/Patch192'
        self.C0_dataset = C0_TrainSet(dir, 35)
        self.LGE_dataset = LGE_TrainSet(dir, 40, train)

        self.len_C0_dataset = self.C0_dataset.__len__()
        self.len_LGE_dataset = self.LGE_dataset.__len__()

        # print(self.len_C0_dataset)
        # print(self.len_LGE_dataset)

    def __len__(self):
        return self.len_C0_dataset if self.len_C0_dataset > self.len_LGE_dataset else self.len_LGE_dataset

    def __getitem__(self, item):
        source_image, source_label = self.C0_dataset.__getitem__(np.random.randint(low=0, high=self.len_C0_dataset))
        target_image, target_label = self.LGE_dataset.__getitem__(item)

        return source_image, source_label, target_image, target_label


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    from torch.utils.data import Dataset, DataLoader

    path = '/data02/imucs_data/machine58_img/wangxiaoming/dataset/MMWHS/Patch192'
    train = False

    SourceData = LGE_TrainSet(path, 5, train)
    # SourceData = Train_Dataset(path, train)
    SourceData_loader = DataLoader(SourceData, batch_size=12, shuffle=True, num_workers=2, pin_memory=True,
                                   drop_last=True)
    print(SourceData.__len__())
    # for i in SourceData_loader:
    #     img, label = i
    #     print(img.shape)  # 12x1x160x160
    #     print(label.shape)  # 12x160x160

    '''
    数据读取一大缺陷：2D、没有增强
    
    数量：
    C0：35
    LGE,vali:5
    LGE,test:40
    
    预训练：C0数据用作预训练
    训练：C0与LGE的image组成对来进行训练，与C0的标签进行比较
    验证：LGE的image与label进行比较
    测试：LGE的test数据集中image与label进行比较
    '''

    # 数据集的分配可能有点问题，训练集少于验证集
