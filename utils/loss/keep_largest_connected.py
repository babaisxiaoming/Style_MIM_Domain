import numpy as np
import torch
from skimage import measure
from monai import transforms

'''
最大联通域，可提高acc
输入输出皆为one-hot形式
'''


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    mask = mask.cpu().detach().numpy()
    num_channel = mask.shape[1]
    out_img = np.zeros(mask.shape, dtype=np.uint8)
    for struc_id in range(1, num_channel + 1):
        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)  # 查找联通域
        props = measure.regionprops(blobs)  # 操作前先转换为此类
        if not props:
            continue
        area = [ele.area for ele in props]  # 计算每个联通域的面积
        largest_blob_ind = np.argmax(area)  # 最大联通域下标
        largest_blob_label = props[largest_blob_ind].label  # 最大联通域下标对应的标记区域

        out_img[blobs == largest_blob_label] = struc_id  # 将当前面积最大区域赋值为struc_id 也就是对应的类

    out_img = torch.from_numpy(out_img).cuda()

    return out_img


def keep_largest_connected_components_monai(mask, num_class=4):
    out = torch.zeros(mask.shape, dtype=mask.dtype).cuda()
    for struc_id in range(1, num_class + 1):
        binary_img = mask == struc_id
        mask_n = transforms.utils.get_largest_connected_component_mask(binary_img, connectivity=1)
        out[mask_n] = struc_id
    return out
