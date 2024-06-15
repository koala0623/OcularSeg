import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils1 import test_single_volume
from datasets.dataset_synapse_ori import Synapse_dataset2, RandomGenerator_test
from torchvision import transforms
from utils_metrics import compute_mIoU
import time
from networks1.Efficient_all import M2SNet



parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str,
                    default='ESLD', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='./predictions2', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.1, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_name', type=str, default='R50+ViT-B_16', help='select one vit model')

args = parser.parse_args()


def inference(args, model, save_predict_path, test_save_path=None):
    db_test = Synapse_dataset2(base_dir=args.volume_path, split="test",
                              transform=transforms.Compose(
                                   [RandomGenerator_test(output_size=args.img_size)]))

    print("The length of train set is: {}".format(len(db_test)))

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    ious = []
    total_time = 0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]  image:tensor(1,3,224,224)   label:tensor(1,224,224)
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, save_predict_path, classes=args.num_classes, patch_size=args.img_size,
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        # iteration_time = test_single_volume(image, label, model, save_predict_path, classes=args.num_classes, patch_size=args.img_size,
        #                         test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)

        # metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        # total_time += iteration_time
    avg_time = total_time/len(db_test)
    print("average time is %f seconds"%avg_time)


    metric_list = metric_list / len(db_test)
    # 各类的 dice score 和 housdorff
    for i in range(0, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i][0], metric_list[i][1]))
    # 平均 dice score 和 housdorff
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    # IoU以及P/R/F1
    compute_mIoU(args.label_path, save_predict_path, 4, patch_size=args.img_size, name_classes=None)

    return "Testing Finished!"




if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset

    dataset_config = {
    'ESLD' : {'volume_path': '/sdata/yixin.zhang/program/dataset/ESLD/test/image/',
                'label_path': '/sdata/yixin.zhang/program/dataset/ESLD/test/label/',
                'img_size': [128,256],
                'z_spacing': 1
                },
    'Mobius': {'volume_path': '/sdata/yixin.zhang/program/dataset/MOBIUS_all/test/image/',
                'label_path': '/sdata/yixin.zhang/program/dataset/MOBIUS_all/test/label/',
                # 'img_size': [192,256],
                'img_size': [160,288],
                'z_spacing': 1
                },
    'Ubiris': {'volume_path': '/sdata/yixin.zhang/program/dataset/Ubiris/test/image/',
                'label_path': '/sdata/yixin.zhang/program/dataset/Ubiris/test/label/',
                'img_size': [160,288],
                'z_spacing': 1
                },
    'SBVPI' : {'volume_path': '/sdata/yixin.zhang/program/dataset/SBVPI/test/image/',
                'label_path': '/sdata/yixin.zhang/program/dataset/SBVPI/test/label/',
                'img_size': [160,288],
                'z_spacing': 1
                }
                    }




    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True
    args.img_size = dataset_config[dataset_name]['img_size']
    args.label_path = dataset_config[dataset_name]['label_path']

    # name the same snapshot defined in train script!
    args.exp = dataset_name + str(args.img_size)
    snapshot_path = "./model/{}/{}".format(args.exp, 'unet')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2]+'k'
    snapshot_path = snapshot_path + '_epoch' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed!=1234 else snapshot_path

    net = M2SNet().cuda()

    # 手动添加权重
    snapshot = r'epoch_100.pth'
    
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    
    args.test_save_dir = './predictions_'+ args.dataset
    save_predict_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)

    if args.is_savenii:
        args.test_save_dir = './predictions_2'
        test_saveall_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_saveall_path, exist_ok=True)
    else:
        test_saveall_path = None
    inference(args, net, save_predict_path, test_saveall_path)




