import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from PIL import Image
import copy
import cv2
from torch.nn import functional as F
import os
import time

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0



# 输出彩色预测结果
def test_single_volume(image, label, net, save_predict, classes, patch_size, test_save_path=None, case=None, z_spacing=1):

    iou_label = label.squeeze(0).cpu().detach()   # tensor(224,224)
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()  # array: (3,224,224)  (224,224)
    
    _, x, y = image.shape
    if x != patch_size[0] or y != patch_size[1]:
        #缩放图像符合网络输入
        image = zoom(image, (1,patch_size[0] / x, patch_size[1] / y), order=3)
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()  # tensor: (1,3,224,224)
    net = net.cuda()
    net.eval()
    start = time.time() 
    with torch.no_grad():
        # out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)  # 预测结果 tensor:(224,224)
        out, att = net(input)
        att = att[:,1:2,:,:]
        att = zoom(att, (1,patch_size[0] / x, patch_size[1] / y), order=3)
        out = torch.argmax(torch.softmax(out, dim=1)*(att)).squeeze(0)


    # end = time.time()
    # iteration_time = end - start
        iou_out = out
        out = out.cpu().detach().numpy()                                        # array: (224,224)
        if x != patch_size[0] or y != patch_size[1]:
            #缩放图像至原始大小
            prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            prediction = out  # array: (224,224)
    # return iteration_time


    true_pred = np.repeat(prediction[:, :, np.newaxis], 3, axis=2) # 160,288->160,288,3
    if not os.path.exists(save_predict):
        os.makedirs(save_predict)
    cv2.imwrite(os.path.join(save_predict, case + ".png"), true_pred)  # 保存的预测黑图
   
    color_pred = true_pred
    info_gt = true_pred.shape
    height = info_gt[0]
    width = info_gt[1]
    for h in range(0, height):
        for j in range(0, width):
            (b, g, r) = color_pred[h, j]
            if (b, g, r) == (0, 0, 0):
                color_pred[h, j] = (0, 0, 0)
            if (b, g, r) == (1, 1, 1):
                color_pred[h, j] = (0, 102, 255)
            if (b, g, r) == (2, 2, 2):
                color_pred[h, j] = (0, 255, 255)
            if (b, g, r) == (3, 3, 3):
                color_pred[h, j] = (255, 153, 0)
    color_save = save_predict + "close"
    if not os.path.exists(color_save):
        os.makedirs(color_save)
    cv2.imwrite(os.path.join(color_save, case + ".png"), color_pred)  # 保存的预测黑图



    metric_list = []
    for i in range(0, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # 可视化
    if test_save_path is not None:

        transposed_iamge = np.transpose(image, (1, 2, 0))  # (224,224,3)
        transposed_iamge = cv2.cvtColor(np.array(transposed_iamge), cv2.COLOR_BGR2RGB) 
        label = label * 85         # array 224,224
        prediction = prediction * 85 # array 224,224
        three_channel_label = np.repeat(label[:, :, np.newaxis], 3, axis=2)   # (224,224,3)
        three_channel_prediction = np.repeat(prediction[:, :, np.newaxis], 3, axis=2)  # (224,224,3)
        cat = np.hstack((transposed_iamge, three_channel_label, three_channel_prediction))  # 224,672,3
        cv2.imwrite(test_save_path+'/'+case+'.png', cat)
    # 计算IoU
    # iou = per_class_mIoU(iou_out, iou_label)

    # return iou, metric_list
    return metric_list


def per_class_mIoU(predictions, targets, info_print=False):  # predict & label  （96，128）
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    unique_labels = np.unique(targets)  # 返回一个包含 targets 数组中所有唯一元素的新数组 [0, 1]，并按升序排列
    ious = list()
    for index in unique_labels:
        pred_i = predictions == index
        label_i = targets == index
        intersection = np.logical_and(label_i, pred_i)
        union = np.logical_or(label_i, pred_i)
        iou_score = np.sum(intersection) / np.sum(union)
        iou_score = iou_score if np.isfinite(iou_score) else 0.0
        ious.append(iou_score)
    if info_print:
        print("per-class mIOU: ", ious)
    return np.average(ious)


def structure_loss(pred, mask):  # predict 16,4,256,256   16,1,256,256
    mask_float = mask.float()
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask_float, kernel_size=31, stride=1, padding=15) - mask)  # tensor 16,1,256,256
    wbce = F.cross_entropy(pred, mask.squeeze(1).to(torch.long))  # 使用交叉熵损失函数     tensor: 1.2040 cuda:0
# .squeeze(1)
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))  # torch.Size([16, 1])
    pred = torch.softmax(pred, dim=1)
    # pred = F.sigmoid(pred)  # 使用softmax将预测结果转换为概率分布   16,4,256,256
    inter = ((pred * mask) * weit).sum(dim=(2, 3))  # 16，4
    union = ((pred + mask) * weit).sum(dim=(2, 3))  # 16，4
    wiou = 1 - (inter + 1) / (union - inter + 1)  # 16，4
    return (wbce + wiou).mean()


# def convert_to_one_hot(labels, num_classes):
#     batch_size, height, width = labels.size()
#     one_hot = torch.zeros(batch_size, num_classes, height, width)
#     for b in range(batch_size):
#         for i in range(height):
#             for j in range(width):
#                 label = labels[b, i, j]
#                 one_hot[b, label-1, i, j] = 1
#     return one_hot

def convert_to_one_hot(labels, num_classes):
    batch_size, height, width = labels.size()
    one_hot = torch.nn.functional.one_hot(labels.long(), num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    return one_hot



def generation2(image, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):

    image = image.squeeze(0).cpu().detach().numpy()  # array: (3,224,224)  (224,224)
    
    _, x, y = image.shape
    if x != patch_size[0] or y != patch_size[1]:
        #缩放图像符合网络输入
        image = zoom(image, (1,patch_size[0] / x, patch_size[1] / y), order=3)
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()  # tensor: (1,3,224,224)
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)  # 预测结果 tensor:(224,224)
        iou_out = out
        out = out.cpu().detach().numpy()                                        # array: (224,224)
        out = out * 85 # array 224,224
        # out = np.transpose(out, (1, 2, 0))
        three_channel_label = np.repeat(out[:, :, np.newaxis], 3, axis=2)   # (224,224,3)

        resized_image = cv2.resize(three_channel_label, (500, 400), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(test_save_path+'/' + case[0] +'.png', resized_image)

