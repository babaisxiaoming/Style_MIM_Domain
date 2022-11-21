import os
import datetime
from utils import write_config, update_config


def create_path(args, exist_root=None):
    save_path = "{}/{}_{}".format(args.result_root_path, args.model_name, args.dataset_name)
    if exist_root is None:
        now = datetime.datetime.now()
        format_now = now.strftime('%y%m%d%H%M%S')

        ckpt_path = os.path.join('{}/{}/ckpt/'.format(save_path, format_now))
        logger_path = os.path.join('{}/{}/logs/'.format(save_path, format_now))
        config_path = os.path.join('{}/{}/{}'.format(save_path, format_now, args.config_name))
        visula_path = os.path.join('{}/{}/{}'.format(save_path, format_now, 'visual'))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        if not os.path.exists(logger_path):
            os.makedirs(logger_path)

        if not os.path.exists(visula_path):
            os.makedirs(visula_path)

        write_config(config_path, args)
    else:

        save_path = os.path.join(save_path, exist_root)
        assert os.path.exists(save_path)
        # print(save_path)
        ckpt_path = os.path.join(save_path, 'ckpt')
        logger_path = os.path.join(save_path, 'logs')
        config_path = os.path.join(save_path, 'transfuse.yaml')
        visula_path = os.path.join(save_path, 'visual')

    return ckpt_path, logger_path, config_path, visula_path


def getresume_path(args):
    root_path = os.path.abspath(os.path.join(os.getcwd()))
    root_path = '/data02/WeiHX_GRP/WhxStuE/wxm/project/demo1'
    print(root_path)
    save_path = "{}/{}/{}_{}".format(root_path, args.result_root_path, args.model_name, args.dataset_name)
    print(f'缓存读取路径：{save_path}')
    ckpt_list = os.listdir(save_path)
    print(f'缓存文件：{ckpt_list}')
    ckpt_list.sort(key=lambda fn: os.path.getmtime(save_path + '/' + fn))
    print(f'最新文件为：{ckpt_list[-1]}')

    resume_path = os.path.join(save_path, ckpt_list[-1], 'ckpt')
    assert os.path.exists(resume_path)
    print(resume_path)
    return resume_path


if __name__ == '__main__':
    from config.model_config import get_config

    args = get_config()

    create_path(args, exist_root='220930124600')
