import torch
import torch.nn as nn
import numpy as np
from monai.networks import one_hot
from monai import transforms
from monai.metrics import DiceMetric

import torch.nn.functional as F
from monai import data
from datasets.MS_CMRSeg_PNG.val_LGE_raw_2 import LGE_SET
from datasets.MS_CMRSeg_PNG.train_bSSFP_LGE import TrainSet
from models.prior_pretain_VAE.VAE import VAE

import matplotlib.pyplot as plt
from monai.losses import DiceLoss, DiceCELoss

ce_loss = nn.CrossEntropyLoss(reduction='none')
dice_loss = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-6, reduction='none')

pred = torch.randn(1, 4, 224, 224)
t_pseudo_label = torch.randint(0, 4, (1, 4, 224, 224)).to(torch.float32)
ce_loss = ce_loss(torch.softmax(pred, dim=1), t_pseudo_label)
print(ce_loss.shape)
# loss_dice = dice_loss(pred, torch.argmax(t_pseudo_label, dim=1).unsqueeze(1))
# print(loss_dice)

print('这是一次改动测试')