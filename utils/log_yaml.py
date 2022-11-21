import yaml
import datetime


def read_config(yamlpath):
    with open(yamlpath, 'r') as f:
        default_arg = yaml.safe_load(f)
    return default_arg


def write_config(yamlpath, args):
    config = {'default_config': {k: v for k, v in vars(args).items()},
              'config': {k: v for k, v in vars(args).items()}}
    with open(yamlpath, "w", encoding="utf-8") as f:
        yaml.dump(config, f)


def update_config(yamlpath, config, parser):
    # 更新config
    def_config = read_config(yamlpath)
    for k, v in config.items():
        def_config['config'][k] = v
    with open(yamlpath, "w", encoding="utf-8") as f:
        yaml.dump(def_config, f)

    # 写入更新信息
    with open(yamlpath, "a", encoding="utf-8") as f:
        now = datetime.datetime.now()
        format_now = now.strftime('%y%m%d%H%M%S')
        updata_title = f'updata_{format_now}'

        updata_config = {updata_title: {k: v for k, v in config.items()}}
        yaml.dump(updata_config, f, encoding='utf-8', allow_unicode=True)

    # 更新parser
    parser.set_defaults(**def_config['config'])
    args = parser.parse_args()
    return args
