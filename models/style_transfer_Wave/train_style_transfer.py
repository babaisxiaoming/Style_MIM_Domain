import argparse
from datetime import datetime, timedelta
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import os
from torch import save

from model import Wave_Model
from train_bSSFP_LGE import TrainSet

from monai import data

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

print("Device name: {}".format(torch.cuda.get_device_name(0)))
start_time = datetime.now()
max_duration = 24 * 3600 - 5 * 60
print("torch version: {}".format(torch.__version__))
print("device count: {}".format(torch.cuda.device_count()))
print('device name: {}'.format(torch.cuda.get_device_name(0)))

cudnn.benchmark = True

torch.autograd.set_detect_anomaly(True)


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str, default='bssfp',
                    help='the modality for content')  # bssfp, t2 or lge
parser.add_argument('--style', type=str, default='lge',
                    help='the modality for content')

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--ns', type=int, default=700)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--augmentation', default=True, action='store_true')
parser.add_argument('--lw', action='store_true')
parser.add_argument('--crop', type=int, default=224)
parser.add_argument('--vgg', help='the path to the directory of the weight', type=str,
                    default='/data02/imucs_data/machine58_img/wangxiaoming/pre_weights/FUDA/vgg_normalised.pth')
parser.add_argument('--style_weight', type=float, default=5.0)
parser.add_argument('--content_weight', type=float, default=5.0)
parser.add_argument('--latent_weight', type=float, default=1.0)
parser.add_argument('--recons_weight', type=float, default=1.0)
parser.add_argument('--save_every_epochs', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()
print('weight_dir: {}'.format(args.vgg))


def get_appendix():
    appendix = args.content + "2" + args.style + '.lr{}'.format(args.lr) + \
               '.sw{}'.format(args.style_weight) + '.cw{}'.format(args.content_weight) + \
               '.lw{}'.format(args.latent_weight) + '.rw{}'.format(args.recons_weight)
    if args.augmentation:
        appendix += '.aug'
    return appendix


appendix = get_appendix()
print(appendix)
device = torch.device('cuda')


def main():
    start_epoch = 0
    print('start epoch: {}'.format(start_epoch))
    network = Wave_Model()
    network.train()
    network.to(device)

    train_dataset = TrainSet("/data02/imucs_data/machine58_img/wangxiaoming/dataset/MS_CMRSeg2019")
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=True,
                                   pin_memory=torch.cuda.is_available())

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    best_loss = 1000
    loss_de = 0
    for i in tqdm(range(start_epoch, args.epochs)):
        loss_c_list, loss_s_list, loss_l_list, loss_r_list = [], [], [], []

        adjust_learning_rate(optimizer, iteration_count=i)

        for content_images, _, _, style_images, _, _ in train_loader:
            content_images = content_images.cuda()
            style_images = style_images.cuda()
            loss_c, loss_s = network(content_images, style_images)

            # collect losses
            loss_c_list.append(loss_c.item())
            loss_s_list.append(loss_s.item())

            optimizer.zero_grad()
            loss_de = loss_c + loss_s
            loss_de.backward()
            optimizer.step()

            # print("{}, {}, {}".format("0:>12".format('epoch: {}'.format(i)), loss_c.item(), loss_s.item()))

        if best_loss > loss_de:
            model_name = '/data02/GongZ_GRP/GzStuA/wxm/Style_MIM_Domain/models/style_transfer_Wave/best_model.pth'
            save({'epoch': i, 'model_state_dict': network.state_dict()}, model_name)


if __name__ == '__main__':
    main()
    print('Time elapsed: {}'.format(datetime.now() - start_time))
    print('program finish')
