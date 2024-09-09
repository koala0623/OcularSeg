import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils1 import DiceLoss, structure_loss
from networks1.M2Snet_ori import LossNet
from torchvision import transforms
# from datasets.dataset_synapse_prior import Synapse_dataset2, RandomGenerator
from datasets.dataset_pupil import Synapse_dataset2, RandomGenerator
import torch.nn.functional as F
from utils1 import convert_to_one_hot
from evaluate import evaluate
from networks1.M2Snet_ori import LossNet
from PIL import Image
from networks1 import settings

def trainer_synapse(args, model, snapshot_path):
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    # Initialize logging
    experiment = wandb.init(project='General_MOBIUS', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.base_lr,
             val_percent=0.1)
    )

    db_train = Synapse_dataset2(base_dir=args.root_path, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=args.img_size)]))
    val_set = Synapse_dataset2(base_dir=args.val_path, split="test",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=args.img_size)]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(2)
    net_loss = LossNet().cuda()
    # msssim_loss = MSSSIM().cuda()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.5, 0.999))

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # len训练数据集中的批次数量 max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            # image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch, fill_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['fill']
            image_batch, label_batch, fill_batch = image_batch.cuda(), label_batch.cuda(), fill_batch.cuda() # image_batch = torch.Size([24, 3, 224, 224])
            
            ####SMSSIM###########
            one_hot_labels = convert_to_one_hot(label_batch, num_classes).cuda()
            #####################
            
            output_4, mu, output_2 = model(image_batch)
            # lateral_map_4, lateral_map_3, lateral_map_2= model(image_batch)
            with torch.no_grad():
                mu = mu.mean(dim=0, keepdim=True)  # mu: 1, 512, 64
                momentum = settings.EM_MOM
                model.emau.mu *= momentum
                model.emau.mu += mu * (1 - momentum)
            #####Lossnet#####
            for param in net_loss.parameters():
                param.requires_grad = False

            loss1u = structure_loss(output_4, label_batch.unsqueeze(dim=1))
            loss2u = net_loss(torch.softmax(output_4[:, 1:, :, :], dim=1), one_hot_labels[:, 1:, :, :])  # 1,1,352,352  1,352,352
            loss_ce = ce_loss(output_2, fill_batch[:].long())  # tensor: 48,4,224,224(logits)     tensor: 48,224,224
            loss_dice = dice_loss(output_2, fill_batch, softmax=True)
            # if epoch_num <= 30:
            #     loss = loss1u + 0.1 * loss2u + loss_ce + loss_dice
            # else:
            #     loss = loss1u + 0.1 * loss2u + 0.5*loss_ce
            loss = loss1u + 0.1 * loss2u + 1*loss_ce + 1*loss_dice
            ###################


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            experiment.log({
                'train loss': loss.item(),  # 累积当前轮次的损失值
                'lossnet':loss2u.item(),
                'structure loss':loss1u.item(),
                'loss_ce':loss_ce.item(),
                'loss_dice':loss_dice.item(),
                'iter': iter_num,
                'epoch': epoch_num+1
            })

            logging.info('iteration %d : loss2u : %f, loss_1u: %f' % (iter_num, loss2u.item(), loss1u.item()))

        # validation
        # if iter_num % 1 == 0:
        logging.info('epoch %d validation starting...' % (epoch_num))
        val_score, IoU = evaluate(model, val_loader, args.amp)
        val_score = val_score * 100
        IoU = IoU * 100
        try:
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation Dice': val_score,
                'validation MIoU': IoU,
                'images': wandb.Image(image_batch[0].cpu()),   # 24,3,224,224  -> 3,224,224
                'masks': {
                    'true': wandb.Image(label_batch[0].float().cpu()),  # 24,224,224  -> 224,224
                    'pred': wandb.Image(output_4.argmax(dim=1)[0].float().cpu()),  # 24,4,224,224 -> 224,224
                    'pred2': wandb.Image(output_2.argmax(dim=1)[0].float().cpu()),  # 24,4,224,224 -> 224,224
                },
                'iter': iter_num,
                'epoch': epoch_num+1
                # **histograms
            })
        except:
            pass

        save_interval = 10 # int(max_epoch/6)
        # if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num+1) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        elif (epoch_num + 1) > 60 :
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num+1) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

    return "Training Finished!"