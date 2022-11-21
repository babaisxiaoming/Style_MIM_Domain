import argparse
import os
import numpy as np

import torch
from torch.autograd import Variable
import tqdm
from torch.utils.data import DataLoader

import torch.backends.cudnn as cudnn
import random

from datasets.MS_CMRSeg_PNG.target_LGE_raw import LGE_SET
from models.convnext_up import ConvNeXt_Up

from models.generate_pseudo.generate_pseudo_function import generate_pseudo

bceloss = torch.nn.BCELoss()
seed = 3377
savefig = False
get_hd = False
if True:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('-g', '--gpu', type=int, default=3)
    parser.add_argument('--save_path',
                        default='/data02/GongZ_GRP/GzStuA/wxm/Style_MIM_Domain/datasets/pseudo_MS_CMRSeg')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # 1. dataset
    db_train = LGE_SET("/data02/imucs_data/machine58_img/wangxiaoming/dataset/MS_CMRSeg2019")
    train_loader = DataLoader(db_train, batch_size=args.batchsize, shuffle=False, num_workers=1)

    # 2. model
    model = ConvNeXt_Up(in_channels=3, out_channels=4)
    ckpt_path = '/data02/GongZ_GRP/GzStuA/wxm/Style_MIM_Domain/result/Cross_sa_vit_unetr_MS_CMRSeg'
    resume_checkpoint_path = os.path.join(ckpt_path, "epoch=890-val_mean_dice=0.744757.ckpt")
    ckpt = torch.load(resume_checkpoint_path)['state_dict']
    new_parms = {}
    for k, v in ckpt.items():
        if k.startswith('seg_model'):
            k = k[10:]
            if not k.startswith('MIM_model'):
                new_parms[k] = v
    model.load_state_dict(new_parms)
    model = model.cuda()
    model.train()

    pseudo_label_dic = {}
    mask_dic = {}

    with torch.no_grad():
        for batch_idx, (sample) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), ncols=80, leave=False):
            data, target, img_name = sample
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            preds = torch.zeros([10, data.shape[0], 4, data.shape[2], data.shape[3]]).cuda()
            features = torch.zeros([10, data.shape[0], 1024, 14, 14]).cuda()
            for i in range(10):
                with torch.no_grad():
                    preds[i, ...], features[i, ...] = model(data)

            pseudo_label, mask = generate_pseudo(preds, features, num_class=4)

            pseudo_label = pseudo_label.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            for i in range(data.shape[0]):
                pseudo_label_dic[img_name[i]] = pseudo_label[i]
                mask_dic[img_name[i]] = mask[i]

        np.savez(f'{args.save_path}/target_LGE_pseudo_right', pseudo_label_dic, mask_dic)
