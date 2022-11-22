import os

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import *
from pytorch_lightning.strategies.ddp import DDPStrategy

from pl_dataset import MyDataset
from pl_interface_stage1 import Interface
from config.style_uda import get_config
from utils import update_config, create_path, getresume_path

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def main(args):
    seed_everything(args.seed, workers=True)

    # ------ get pl_dataset ------
    data_module = MyDataset(args)

    # ------ set save path ------
    exist_root = None
    ckpt_path, logger_path, config_path, visula_path = create_path(args, exist_root=exist_root)

    # ------ logger ------
    wandb_logger = WandbLogger(save_dir=logger_path,
                               offline=False,
                               project="{}_{}".format(args.model_name, args.dataset_name))

    # ------ checkpoint ------
    early_stop = EarlyStopping(monitor="val_loss",
                               mode="min",
                               patience=10,
                               check_on_train_epoch_end=False)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint = ModelCheckpoint(monitor="val_mean_dice",
                                 dirpath=ckpt_path,
                                 filename='{epoch:03d}-{val_mean_dice:.6f}',
                                 mode='max',
                                 save_top_k=1,
                                 save_last=True,
                                 save_weights_only=False)

    callbacks = [checkpoint, lr_monitor]

    # ------ train ------
    if train:
        data_module.setup("fit")
        trainer = Trainer(callbacks=callbacks,
                          logger=wandb_logger,
                          max_epochs=args.max_epoch,
                          gpus=args.gpus,
                          strategy="ddp",
                          check_val_every_n_epoch=args.check_val_every_n_epoch,
                          num_sanity_val_steps=args.num_sanity_val_steps,
                          gradient_clip_val=args.gradient_clip_val,
                          log_every_n_steps=args.log_every_n_steps,  # 频繁写入会影响速度
                          # auto_lr_find=True,
                          # auto_scale_batch_size='binsearch',
                          precision=16  # 半精度，降低显存
                          )

        model = Interface(args=args, wandb_logger=wandb_logger, visula_path=visula_path)

        if resume:
            resume_checkpoint_path = ''
            model = Interface.load_from_checkpoint(resume_checkpoint_path, args=args)

        # trainer.tune(model, datamodule=data_module)  # auto_lr_find:0.01445439770745928与auto_scale_batch_size:1000
        trainer.fit(model, datamodule=data_module)

    # ------ test ------
    if test:
        data_module.setup("test")
        ckpt_path = ''
        trainer = Trainer(logger=wandb_logger, gpus=1)
        model = Interface(args=args, visula_path=visula_path)
        trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)

    wandb.finish()


if __name__ == '__main__':
    train = True
    resume = False
    test = False

    # ------ combine args ------
    args = get_config()

    main(args)
