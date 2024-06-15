import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from networks1.two_seg import M2SNet, LossNet

import sys
sys.path.insert(0, '/sdata/yixin.zhang/program/pythonfolder/M2S_mobilenet/trainer1/')
from trainer_all import trainer_synapse


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str,
                    default='ESLD', help='root dir for data')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.1,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=[160,288], help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--is_pretrain', type=bool, 
                    default=True, help='random seed')
parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')


args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True  # 使得CuDNN根据输入数据的大小和格式自动选择最适合的算法，从而提高计算的效率
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False  # 确保每次运行网络时的输出结果是确定的
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'ESLD' : {'root_path': '/sdata/yixin.zhang/program/dataset/ESLD/train/image/',
            'val_path': '/sdata/yixin.zhang/program/dataset/ESLD/test/image/',
            'img_size': [128,256]  
                },
        'Mobius': {'root_path': '/sdata/yixin.zhang/program/dataset/MOBIUS_all/train/image/',
                   'val_path': '/sdata/yixin.zhang/program/dataset/MOBIUS_all/test/image/',
                   'img_size': [160,288] 
                   },
        'Ubiris': {'root_path': '/sdata/yixin.zhang/program/dataset/Ubiris/train/image/',
                   'val_path': '/sdata/yixin.zhang/program/dataset/Ubiris/test/image/',
                   'img_size': [160,288]
                   },
        'SBVPI' : {'root_path': '/sdata/yixin.zhang/program/dataset/SBVPI/train/image/',
                    'val_path': '/sdata/yixin.zhang/program/dataset/SBVPI/test/image/',
                    'img_size': [160,288]  
                    }
    }

    args.root_path = dataset_config[dataset_name]['root_path']
    args.val_path = dataset_config[dataset_name]['val_path']
    args.img_size = dataset_config[dataset_name]['img_size']

    args.exp = dataset_name + str(args.img_size)
    snapshot_path = "./model/{}/{}".format(args.exp, 'single_periocular')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k'
    snapshot_path = snapshot_path + '_epoch' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)


    # 设置要使用的GPU编号
    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    
    net = M2SNet().cuda()

    trainer = {args.dataset: trainer_synapse}
    trainer[dataset_name](args, net, snapshot_path)
