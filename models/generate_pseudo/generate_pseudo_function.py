import torch
import torch.nn.functional as F


def _channel(pseudo_label, feature, prediction_small, std_map_small, std_map):
    size = feature.size()
    pseudo_label = pseudo_label.unsqueeze(1)
    prediction_small = prediction_small.unsqueeze(1)
    std_map_small = std_map_small.unsqueeze(1)
    std_map = std_map.unsqueeze(1)

    # target
    target_obj = F.interpolate(pseudo_label, size=size[2:], mode='nearest')
    target_bck = 1.0 - target_obj

    # mask
    mask_ori_obj = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
    mask_ori_bck = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
    mask_ori_obj[std_map < 0.05] = 1.0
    mask_ori_bck[std_map < 0.05] = 1.0
    mask_ori = mask_ori_obj * pseudo_label + mask_ori_bck * pseudo_label

    # mask_small
    mask_obj = torch.zeros([size[0], 1, size[2], size[3]]).cuda()
    mask_bck = torch.zeros([size[0], 1, size[2], size[3]]).cuda()
    mask_obj[std_map_small < 0.05] = 1.0
    mask_bck[std_map_small < 0.05] = 1.0
    mask_small = mask_obj + mask_bck

    # feature
    feature_obj = feature * target_obj * mask_obj  # feature可能有点问题
    feature_bck = feature * target_bck * mask_bck

    # centroid
    centroid_obj = torch.sum(feature_obj * prediction_small, dim=[0, 2, 3], keepdim=True)
    centroid_bck = torch.sum(feature_bck * (1.0 - prediction_small), dim=[0, 2, 3], keepdim=True)

    # target_cnt
    target_obj_cnt = torch.sum(mask_obj * target_obj * prediction_small, dim=[0, 2, 3], keepdim=True)
    target_bck_cnt = torch.sum(mask_bck * target_bck * (1.0 - prediction_small), dim=[0, 2, 3], keepdim=True)
    centroid_obj /= target_obj_cnt
    centroid_bck /= target_bck_cnt

    # distance
    distance_obj = torch.sum(torch.pow(feature - centroid_obj, 2), dim=1, keepdim=True)
    distance_bck = torch.sum(torch.pow(feature - centroid_bck, 2), dim=1, keepdim=True)

    # proto_pseudo
    proto_pseudo = torch.zeros([size[0], 1, size[2], size[3]]).cuda()
    proto_pseudo[distance_obj < distance_bck] = 1.0

    return mask_ori, mask_small, proto_pseudo


def generate_pseudo(predictions, features, num_class=4, threshold=0.75):
    '''
    :param predictions: KxBxCxHxW
    :param features: KxBxCxHxW
    :param num_class: 4
    '''
    assert predictions.shape[2] == num_class
    K, B, C, H, W = predictions.size()
    ori_size = (H, W)
    small_size = (features.size()[3], features.size()[4])
    preds1 = torch.softmax(predictions, dim=2)
    preds = torch.softmax(predictions / num_class, dim=2)
    std_map = torch.std(preds, dim=0)
    prediction = torch.mean(preds1, dim=0)  # 8x4x224x224
    feature = torch.mean(features, dim=0)  # 8x1024x14x14

    pseudo_label = prediction.clone()
    pseudo_label[pseudo_label > threshold] = 1.0  # 注意这里阈值的选择
    pseudo_label[pseudo_label <= threshold] = 0.0

    prediction_small = F.interpolate(prediction, size=small_size, mode='bilinear', align_corners=True)
    std_map_small = F.interpolate(std_map, size=small_size, mode='bilinear', align_corners=True)

    mask = torch.zeros([B, num_class, ori_size[0], ori_size[1]]).cuda()
    mask_small = torch.zeros([B, num_class, small_size[0], small_size[1]]).cuda()  # 暂时不知道干嘛的
    proto_pseudo = torch.zeros([B, num_class, small_size[0], small_size[1]]).cuda()
    for c in range(num_class):
        mask[:, c:, :], mask_small[:, c:, :], proto_pseudo[:, c:, :] = _channel(pseudo_label[:, c, :],
                                                                                feature,
                                                                                prediction_small[:, c, :],
                                                                                std_map_small[:, c, :],
                                                                                std_map[:, c, :])
    proto_pseudo = F.interpolate(proto_pseudo, size=ori_size, mode='nearest')

    # mask_proto = torch.zeros([B, num_class, H, W]).cuda()
    # mask_proto[pseudo_label == proto_pseudo] = 1.0
    # mask = mask * mask_proto

    # return pseudo_label, std_map, proto_pseudo
    return pseudo_label, mask


if __name__ == '__main__':
    preds = torch.randn(10, 8, 4, 224, 224).cuda()
    features = torch.randn(10, 8, 1024, 14, 14).cuda()
    # preds = torch.randn(8, 4, 224, 224).cuda()
    # features = torch.randn(8, 1024, 14, 14).cuda()

    res = generate_pseudo(preds, features)
    for i in res:
        print(i.shape)
