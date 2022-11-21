import argparse


def get_config():
    parser = argparse.ArgumentParser(description="segmentation pipeline")
    parser.add_argument('--seed', default=100)
    parser.add_argument('--gpus', default=[0], help="gpu use list or str")
    parser.add_argument("--lr", default=1.5e-2, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--size", default=(-1, 224, 224), type=tuple, help="input image size")
    parser.add_argument("--num_workers", default=8, type=int, help="使用更多的线程保证显存利用率")
    parser.add_argument("--max_epoch", default=1300, type=int)
    parser.add_argument("--num_class", default=4, type=int)
    parser.add_argument('--gradient_clip_val', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='epochs to warmup LR')

    # ------------------------------------------- debug -------------------------------------------
    parser.add_argument("--fast_dev_run", default=True, type=bool, help="runs 1 train,val,test atch")
    parser.add_argument("--weights_summary", default='full', type=str, help="print the weights summary")
    parser.add_argument("--num_sanity_val_steps", default=1, type=int, help="check runs n batch val before starting")

    parser.add_argument("--limit_train_batches", default=1, type=int, help="how many train batch to check")
    parser.add_argument("--limit_val_batches", default=1, type=int, help="how many val batch to validate")
    parser.add_argument("--limit_test_batches", default=1, type=int, help="how many val batch to validate")

    # ------------------------------------------- frequency -------------------------------------------
    parser.add_argument("--log_every_n_steps", default=10, type=int, help="how many train_step to log once")
    parser.add_argument("--check_val_every_n_epoch", default=3, type=int, help="check val every n train epochs")

    # ------------------------------------------- loss rate -------------------------------------------
    parser.add_argument("--consis_rate", default=2e-3, type=int, help="train_consistency_loss")
    parser.add_argument("--prior_rate", default=1.0, type=int, help="train_consistency_loss")

    # ------------------------------------------- optimizer -------------------------------------------
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.5, help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")

    # ------------------------------------------- dataset -------------------------------------------
    parser.add_argument("--data_dir", default='/data02/imucs_data/machine58_img/wangxiaoming/dataset/MS_CMRSeg2019')
    parser.add_argument("--len_data", default=1000, type=int)

    # ------------------------------------------- model -------------------------------------------
    parser.add_argument('--vgg_encoder', type=str,
                        default='/data02/imucs_data/machine58_img/wangxiaoming/pre_weights/FUDA/vgg_normalised.pth')
    parser.add_argument('--vgg_decoder', type=str,
                        default='/data02/imucs_data/machine58_img/wangxiaoming/pre_weights/FUDA/best_decoder.bssfp2t2.lr0.0001.sw5.0.cw5.0.lw1.0.rw5.0.aug.e200.Scr7.691.pt')
    parser.add_argument('--style_encoder', type=str,
                        default='/data02/imucs_data/machine58_img/wangxiaoming/pre_weights/FUDA/best_fc_encoder.bssfp2t2.lr0.0001.sw5.0.cw5.0.lw1.0.rw5.0.aug.e200.Scr7.691.pt')
    parser.add_argument('--style_decoder', type=str,
                        default='/data02/imucs_data/machine58_img/wangxiaoming/pre_weights/FUDA/best_fc_decoder.bssfp2t2.lr0.0001.sw5.0.cw5.0.lw1.0.rw5.0.aug.e200.Scr7.691.pt')

    # ------------------------------------------- result save -------------------------------------------
    parser.add_argument("--result_root_path", default='result', type=str, help='save result root path')
    parser.add_argument("--model_name", default='Cross_sa_vit_unetr', type=str, help='using base_model name')
    parser.add_argument("--dataset_name", default='MS_CMRSeg', type=str, help='pl_dataset name')
    parser.add_argument("--config_name", default='transfuse.yaml', type=str, help='pl_dataset name')

    args = parser.parse_args()
    return args
