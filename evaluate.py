import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
import numpy as np

@torch.inference_mode()
def evaluate(model, dataloader, amp):
    model.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    ious = list()

    # iterate over the validation set
    with torch.autocast('cuda', enabled=amp):
    # 执行需要进行自动混合精度训练的代码块
        for i_batch, sampled_batch in enumerate(dataloader):  # 24,3,224,224  24,224,224   24,224,224
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()  # image_batch = torch.Size([24, 3, 224, 224])    label 24,224,224
            output_l,_ = model(image_batch)  #24,4,224,224
            # convert to one-hot format
            mask_true = F.one_hot(label_batch, 4).permute(0, 3, 1, 2).float()   # 24,4,224,224
            mask_pred = F.one_hot(output_l.argmax(dim=1), 4).permute(0, 3, 1, 2).float()  # 24,4,224,224
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
        
            iou = per_class_mIoU(torch.argmax(torch.softmax(output_l, dim=1), dim=1), label_batch)    
            ious.append(iou)


    model.train()
    return dice_score / max(num_val_batches, 1), np.average(ious)


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def per_class_mIoU(predictions, targets, info=False): 
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    unique_labels = np.unique(targets)  # 去掉重复值
    ious = list()
    for index in unique_labels:
        pred_i = predictions == index
        label_i = targets == index
        intersection = np.logical_and(label_i, pred_i)
        union = np.logical_or(label_i, pred_i)
        iou_score = np.sum(intersection)/np.sum(union)
        iou_score = iou_score if np.isfinite(iou_score) else 0.0
        ious.append(iou_score)
    if info:
        print ("per-class mIOU: ", ious)
    return np.average(ious)