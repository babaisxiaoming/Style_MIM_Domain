import torch
import pytorch_lightning as pl
from monai import data

from datasets.MS_CMRSeg_PNG.train_bSSFP_LGE import TrainSet
# from datasets.MS_CMRSeg_PNG.test_train import TrainSet

from datasets.MS_CMRSeg_PNG.val_LGE_raw_2 import LGE_SET


class MyDataset(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_dir = self.args.data_dir
        self.len_data = self.args.len_data
        # -------------- dataset --------------
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        if stage == "fit":
            self.train_dataset = TrainSet(data_path=self.data_dir, len_data=self.len_data)
            self.val_dataset = LGE_SET(self.data_dir)
        if stage == "test":
            self.test_dataset = LGE_SET(self.data_dir)

    def train_dataloader(self):
        dataloader = data.DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                     num_workers=self.args.num_workers, drop_last=True,
                                     pin_memory=torch.cuda.is_available())
        return dataloader

    def val_dataloader(self):
        dataloader = data.DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False,
                                     num_workers=self.args.num_workers, drop_last=True,
                                     pin_memory=torch.cuda.is_available())
        return dataloader

    def test_dataloader(self):
        dataloader = data.DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False,
                                     num_workers=self.args.num_workers, drop_last=True,
                                     pin_memory=torch.cuda.is_available())
        return dataloader


if __name__ == '__main__':
    from config.style_uda import get_config

    args = get_config()
    data_module = MyDataset(args)
    data_module.setup('fit')

    loader = data_module.train_dataloader()
    for idx, data in enumerate(loader):
        s_data, s_label, s_name, t_data, t_label, t_name, t_pseudo_label, t_mask = data
        print(s_data.shape)
        print(s_label.shape)
        print(s_name)
        print(t_data.shape)
        print(t_label.shape)
        print(t_name)
        print(t_pseudo_label.shape)
        print(t_mask.shape)
