from torchsummary import summary
from torchstat import stat
import torchinfo

'''方法1，自定义函数 参考自 https://blog.csdn.net/qq_33757398/article/details/109210240'''


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 20 + 'weight name' + ' ' * 20 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 5 + 'number' + ' ' * 5 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 50:
            key = key + (50 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (20 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)


def summary_model(model):
    torchinfo.summary(model, (3, 224, 224), batch_dim=0,
                      col_names=('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'), verbose=1)


def stat_model(model):
    stat(model, input_size=(3, 128, 128, 128))
