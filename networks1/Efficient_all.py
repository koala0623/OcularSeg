import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/sdata/yixin.zhang/program/pythonfolder/M2S_efficient/')
from networks1.res2net import res2net50_v1b_26w_4s
import torchvision
from networks1.EMA import EMAU
from networks1 import settings

# class CNN1(nn.Module):
#     def __init__(self,channel,map_size,pad):
#         super(CNN1,self).__init__()
#         self.weight = nn.Parameter(torch.ones(channel,channel,map_size,map_size),requires_grad=False).cuda()
#         self.bias = nn.Parameter(torch.zeros(channel),requires_grad=False).cuda()
#         self.pad = pad
#         self.norm = nn.BatchNorm2d(channel)
#         self.relu = nn.ReLU()

#     def forward(self,x):
#         out = F.conv2d(x,self.weight,self.bias,stride=1,padding=self.pad)
#         out = self.norm(out)
#         out = self.relu(out)
#         return out



# class M2SNet(nn.Module):
#     # res2net based encoder decoder
#     def __init__(self):
#         super(M2SNet, self).__init__()
#         # ---- ResNet Backbone ----
#         self.resnet = res2net50_v1b_26w_4s(pretrained=True)
#         self.conv_3 = CNN1(64,3,1)  # 卷积核数量、卷积核大小和步长
#         self.conv_5 = CNN1(64,5,2)


#         self.x5_dem_1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.x4_dem_1 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.x3_dem_1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.x2_dem_1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.x1_dem_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.x5_x4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

#         self.x5_x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

#         self.x5_x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
#                                          nn.ReLU(inplace=True))
#         self.x5_dem_4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.x5_x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
#                                          nn.ReLU(inplace=True))
#         self.level4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.level3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.level2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.level1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

#         self.emau = EMAU(64, 4, settings.STAGE_NUM)


#         self.x5_dem_5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
#                                       nn.ReLU(inplace=True))
#         self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.output2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
#         self.output1 = nn.Sequential(nn.Conv2d(64, 4, kernel_size=3, padding=1))


#         # self.atten1 = nn.Sequential(nn.Conv2d(64, 2, kernel_size=3, padding=1), nn.BatchNorm2d(2), nn.ReLU(inplace=True))

#     def forward(self, x):                # bs,3,224,224
#         input = x

#         # '''
#         x = self.resnet.conv1(x)         # bs,64,112,112
#         x = self.resnet.bn1(x)
#         x = self.resnet.relu(x)
#         x1 = self.resnet.maxpool(x)      # bs,64,56,56
#         # ---- low-level features ----
#         x2 = self.resnet.layer1(x1)      # bs,256,56,56
#         x3 = self.resnet.layer2(x2)      # bs,512,28,28
#         x4 = self.resnet.layer3(x3)      # bs,1024,14,14
#         x5 = self.resnet.layer4(x4)      # bs,2048,7,7
#         # '''


#         x5_dem_1 = self.x5_dem_1(x5)     # bs,64,7,7
#         x4_dem_1 = self.x4_dem_1(x4)     # bs,64,14,14
#         x3_dem_1 = self.x3_dem_1(x3)     # bs,64,28,28
#         x2_dem_1 = self.x2_dem_1(x2)     # bs,64,56,56
#         x1_dem_1 = self.x1_dem_1(x1)     # bs,64,56,56

#         x5_dem_1, mu1 = self.emau(x5_dem_1)     # mu 2,64,4
#         x4_dem_1, mu2 = self.emau(x4_dem_1)     # mu 2,64,4
#         x3_dem_1, mu3 = self.emau(x3_dem_1)     # mu 2,64,4
#         x2_dem_1, mu4 = self.emau(x2_dem_1)     # mu 2,64,4
#         x1_dem_1, mu5 = self.emau(x1_dem_1)     # mu 2,64,4


#         x5_dem_1_up = F.upsample(x5_dem_1, size=x4.size()[2:], mode='bilinear')     # bs,64,14,14
#         x5_dem_1_up_map1 = self.conv_3(x5_dem_1_up)
#         x4_dem_1_map1 = self.conv_3(x4_dem_1)
#         x5_dem_1_up_map2 = self.conv_5(x5_dem_1_up)
#         x4_dem_1_map2 = self.conv_5(x4_dem_1)
#         x5_4 = self.x5_x4(
#             abs(x5_dem_1_up - x4_dem_1)+abs(x5_dem_1_up_map1-x4_dem_1_map1)+abs(x5_dem_1_up_map2-x4_dem_1_map2))


#         x4_dem_1_up = F.upsample(x4_dem_1, size=x3.size()[2:], mode='bilinear')
#         x4_dem_1_up_map1 = self.conv_3(x4_dem_1_up)
#         x3_dem_1_map1 = self.conv_3(x3_dem_1)
#         x4_dem_1_up_map2 = self.conv_5(x4_dem_1_up)
#         x3_dem_1_map2 = self.conv_5(x3_dem_1)
#         x4_3 = self.x4_x3(
#             abs(x4_dem_1_up - x3_dem_1)+abs(x4_dem_1_up_map1-x3_dem_1_map1)+abs(x4_dem_1_up_map2-x3_dem_1_map2) )


#         x3_dem_1_up = F.upsample(x3_dem_1, size=x2.size()[2:], mode='bilinear')
#         x3_dem_1_up_map1 = self.conv_3(x3_dem_1_up)
#         x2_dem_1_map1 = self.conv_3(x2_dem_1)
#         x3_dem_1_up_map2 = self.conv_5(x3_dem_1_up)
#         x2_dem_1_map2 = self.conv_5(x2_dem_1)
#         x3_2 = self.x3_x2(
#             abs(x3_dem_1_up - x2_dem_1)+abs(x3_dem_1_up_map1-x2_dem_1_map1)+abs(x3_dem_1_up_map2-x2_dem_1_map2) )


#         x2_dem_1_up = F.upsample(x2_dem_1, size=x1.size()[2:], mode='bilinear')
#         x2_dem_1_up_map1 = self.conv_3(x2_dem_1_up)
#         x1_map1 = self.conv_3(x1)
#         x2_dem_1_up_map2 = self.conv_5(x2_dem_1_up)
#         x1_map2 = self.conv_5(x1)
#         x2_1 = self.x2_x1(abs(x2_dem_1_up - x1)+abs(x2_dem_1_up_map1-x1_map1)+abs(x2_dem_1_up_map2-x1_map2) )


#         x5_4_up = F.upsample(x5_4, size=x4_3.size()[2:], mode='bilinear')
#         x5_4_up_map1 = self.conv_3(x5_4_up)
#         x4_3_map1 = self.conv_3(x4_3)
#         x5_4_up_map2 = self.conv_5(x5_4_up)
#         x4_3_map2 = self.conv_5(x4_3)
#         x5_4_3 = self.x5_x4_x3(abs(x5_4_up - x4_3) +abs(x5_4_up_map1-x4_3_map1)+abs(x5_4_up_map2-x4_3_map2))


#         x4_3_up = F.upsample(x4_3, size=x3_2.size()[2:], mode='bilinear')
#         x4_3_up_map1 = self.conv_3(x4_3_up)
#         x3_2_map1 = self.conv_3(x3_2)
#         x4_3_up_map2 = self.conv_5(x4_3_up)
#         x3_2_map2 = self.conv_5(x3_2)
#         x4_3_2 = self.x4_x3_x2(abs(x4_3_up - x3_2)+abs(x4_3_up_map1-x3_2_map1)+abs(x4_3_up_map2-x3_2_map2) )


#         x3_2_up = F.upsample(x3_2, size=x2_1.size()[2:], mode='bilinear')
#         x3_2_up_map1 = self.conv_3(x3_2_up)
#         x2_1_map1 = self.conv_3(x2_1)
#         x3_2_up_map2 = self.conv_5(x3_2_up)
#         x2_1_map2 = self.conv_5(x2_1)
#         x3_2_1 = self.x3_x2_x1(abs(x3_2_up - x2_1)+abs(x3_2_up_map1-x2_1_map1)+abs(x3_2_up_map2-x2_1_map2) )


#         x5_4_3_up = F.upsample(x5_4_3, size=x4_3_2.size()[2:], mode='bilinear')
#         x5_4_3_up_map1 = self.conv_3(x5_4_3_up)
#         x4_3_2_map1 = self.conv_3(x4_3_2)
#         x5_4_3_up_map2 = self.conv_5(x5_4_3_up)
#         x4_3_2_map2 = self.conv_5(x4_3_2)
#         x5_4_3_2 = self.x5_x4_x3_x2(
#             abs(x5_4_3_up - x4_3_2)+abs(x5_4_3_up_map1-x4_3_2_map1)+abs(x5_4_3_up_map2-x4_3_2_map2) )


#         x4_3_2_up = F.upsample(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear')
#         x4_3_2_up_map1 = self.conv_3(x4_3_2_up)
#         x3_2_1_map1 = self.conv_3(x3_2_1)
#         x4_3_2_up_map2 = self.conv_5(x4_3_2_up)
#         x3_2_1_map2 = self.conv_5(x3_2_1)
#         x4_3_2_1 = self.x4_x3_x2_x1(
#             abs(x4_3_2_up - x3_2_1) +abs(x4_3_2_up_map1-x3_2_1_map1)+abs(x4_3_2_up_map2-x3_2_1_map2))


#         x5_dem_4 = self.x5_dem_4(x5_4_3_2)
#         x5_dem_4_up = F.upsample(x5_dem_4, size=x4_3_2_1.size()[2:], mode='bilinear')
#         x5_dem_4_up_map1 = self.conv_3(x5_dem_4_up)
#         x4_3_2_1_map1 = self.conv_3(x4_3_2_1)
#         x5_dem_4_up_map2 = self.conv_5(x5_dem_4_up)
#         x4_3_2_1_map2 = self.conv_5(x4_3_2_1)
#         x5_4_3_2_1 = self.x5_x4_x3_x2_x1(
#             abs(x5_dem_4_up - x4_3_2_1)+abs(x5_dem_4_up_map1-x4_3_2_1_map1)+abs(x5_dem_4_up_map2-x4_3_2_1_map2) )

#         level4 = self.level4(x4_dem_1 + x5_4)                                   # 2,64,14,14
#         level3 = self.level3(x3_dem_1 + x4_3 + x5_4_3)                          # 2,64,28,28
#         level2 = self.level2(x2_dem_1 + x3_2 + x4_3_2 + x5_4_3_2)               # 2,64,56,56
#         level1 = self.level1(x1_dem_1 + x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1)  # 2,64,56,56




#         # x5_dem_5 = self.x5_dem_5(x5)   # 2,2048,7,7  ->  2,64,7,7
#         output4 = self.output4(F.upsample(x5_dem_1,size=level4.size()[2:], mode='bilinear') + level4)  # 2,64,14,14
#         output3 = self.output3(F.upsample(output4,size=level3.size()[2:], mode='bilinear') + level3)  # 2,64,28,28

#         # attention
#         # atten = self.atten1(F.upsample(output3, size=level2.size()[2:], mode='bilinear'))   # 2,2,56,56
#         # atten1 = torch.softmax(atten, dim=1)  # 2,1,28,28 
#         # atten2 = atten1[:, 1:2, :, :]
#         output2 = self.output2(F.upsample(output3,size=level2.size()[2:], mode='bilinear') + level2)
#         # output2 = output2 * atten2


#         output1 = self.output1(F.upsample(output2,size=level1.size()[2:], mode='bilinear') + level1)
#         output = F.upsample(output1, size=input.size()[2:], mode='bilinear')

#         mu = mu1 + mu2 + mu3 + mu4 + mu5
#         if self.training:
#             return output,mu
#         return output
    
    
# class LossNet(torch.nn.Module):
#     def __init__(self, resize=False):
#         super(LossNet, self).__init__()
#         blocks = []
#         blocks.append(torchvision.models.vgg16(weights="IMAGENET1K_V1").features[:4].eval())
#         blocks.append(torchvision.models.vgg16(weights="IMAGENET1K_V1").features[4:9].eval())
#         blocks.append(torchvision.models.vgg16(weights="IMAGENET1K_V1").features[9:16].eval())
#         blocks.append(torchvision.models.vgg16(weights="IMAGENET1K_V1").features[16:23].eval())
#         for bl in blocks:
#             for p in bl:
#                 p.requires_grad = False
#         self.blocks = torch.nn.ModuleList(blocks)
#         self.transform = torch.nn.functional.interpolate
#         self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
#         self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
#         self.resize = resize

#     def forward(self, input, target):  # 24,352,352   24,352,352
#         if input.shape[1] != 3:
#             input = input.unsqueeze(1).expand(input.shape[0], 3, 224, 224)
#             target = target.unsqueeze(1).expand(input.shape[0], 3, 224, 224)  # 1,3,352,352
#         input = (input-self.mean) / self.std   # 1,3,1,1   1,3,1,1  1,3,352,352
#         target = (target-self.mean) / self.std  
#         if self.resize:
#             input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)   # 1,3,224,224
#             target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False) # 1,3,224,224
#         loss = 0.0
#         x = input
#         y = target

#         for block in self.blocks:
#             x = block(x)  # 1,64,224,224
#             y = block(y)  # 1,64,224,224
#             loss += torch.nn.functional.mse_loss(x, y)
#         return loss

# # if __name__ == '__main__':
# #     model = M2SNet().cuda()
# #     x = torch.rand(2, 3, 224,224).cuda()
# #     output4, output2, mu = model(x)
# #     print(output4.shape)  # 2,2,224,224
# #     print(output2.shape)  # 2,2,224,224
# #     print(mu.shape)
    
# if __name__ == '__main__':
#     model = M2SNet().cuda()
#     x = torch.rand(2, 3, 224,224).cuda()
#     output4= model(x)
#     print(output4.shape)  # 2,2,224,224





import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/sdata/yixin.zhang/program/pythonfolder/M2S_efficient/')
# from networks1.res2net import res2net50_v1b_26w_4s
import torchvision
from PIL import Image
import numpy as np
import timm

class CNN1(nn.Module):
    def __init__(self,channel,map_size,pad):
        super(CNN1,self).__init__()
        self.weight = nn.Parameter(torch.ones(channel,channel,map_size,map_size),requires_grad=False).cuda()
        self.bias = nn.Parameter(torch.zeros(channel),requires_grad=False).cuda()
        self.pad = pad
        self.norm = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = F.conv2d(x,self.weight,self.bias,stride=1,padding=self.pad)
        out = self.norm(out)
        out = self.relu(out)
        return out



class M2SNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self):
        super(M2SNet, self).__init__()
        # ---- ResNet Backbone ----
        self.efficient = timm.create_model('efficientnetv2_rw_s.ra2_in1k', pretrained=False)
        state_dict = torch.load('/sdata/yixin.zhang/program/pythonfolder/M2S_efficient/pth/efficientnet_v2s_ra2_288-a6477665.pth')
        self.efficient.load_state_dict(state_dict)
        self.conv_3 = CNN1(64,3,1)  # 卷积核数量、卷积核大小和步长
        self.conv_5 = CNN1(64,5,2)


        self.x5_dem_1 = nn.Sequential(nn.Conv2d(272, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(160, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(48, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x1_dem_1 = nn.Sequential(nn.Conv2d(24, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x5_x4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.x5_x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.x5_x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.x5_dem_4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x5_x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.level4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.emau = EMAU(64, 4, settings.STAGE_NUM)

        self.x5_dem_5 = nn.Sequential(nn.Conv2d(272, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(64, 4, kernel_size=3, padding=1))
        self.atten1 = nn.Sequential(nn.Conv2d(64, 2, kernel_size=3, padding=1), nn.BatchNorm2d(2), nn.ReLU(inplace=True))

        self.transposed_conv = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)


        # self.t1 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=3, padding=1))

    def forward(self, x):                # bs,3,224,224
        input = x

        # '''
        x = self.efficient.conv_stem(x)         # 2,16,112,112           2,48,112,112
        x = self.efficient.bn1(x)                   # res2net50     effi-b4           effi-v2
        x1 = self.efficient.blocks[0](x)         # 2,16,112,112    2,24,112,112       2,24,112,112
        x2 = self.efficient.blocks[1](x1)         # 2,24,56,56     2,32,56,56         2,48,56,56
        x3 = self.efficient.blocks[2](x2)         # 2,40,28,28     2,56,28,28         2,64,28,28
        x4 = self.efficient.blocks[3](x3)         # 2,80,14,14     2,112,14,14        2,160,14,14
        x4 = self.efficient.blocks[4](x4)         # 2,112,14,14    2,160,14,14        2,160,14,14
        x5 = self.efficient.blocks[5](x4)         # 2,160,7,7      2,272,7,7          2,272,7,7


        x5_dem_1 = self.x5_dem_1(x5)     # bs,64,7,7
        x4_dem_1 = self.x4_dem_1(x4)     # bs,64,14,14
        x3_dem_1 = self.x3_dem_1(x3)     # bs,64,28,28
        x2_dem_1 = self.x2_dem_1(x2)     # bs,64,56,56
        x1_dem_1 = self.x1_dem_1(x1)     # bs,64,56,56       16,64,32,64

        # t1 = self.t1(x1_dem_1)        #  16,3,32,64
        # t1 = F.upsample(t1, size=(128,256), mode='bilinear')   # 16,3,128,256
        # array = torch.softmax(t1, dim = 1)[0].cpu().detach().numpy()
        # image = Image.fromarray((array*255).astype(np.uint8).transpose(1,2,0))
        # image.save("image1.jpg")



        x5_dem_1, mu1 = self.emau(x5_dem_1)     # mu 2,64,4
        x4_dem_1, mu2 = self.emau(x4_dem_1)     # mu 2,64,4
        x3_dem_1, mu3 = self.emau(x3_dem_1)     # mu 2,64,4
        x2_dem_1, mu4 = self.emau(x2_dem_1)     # mu 2,64,4
        x1_dem_1, mu5 = self.emau(x1_dem_1)     # mu 2,64,4


        x5_dem_1_up = F.upsample(x5_dem_1, size=x4.size()[2:], mode='bilinear')     # bs,64,14,14
        x5_dem_1_up_a3 = F.upsample(x5_dem_1, size=x3.size()[2:], mode='bilinear')     # bs,64,28,28  ***
        x5_dem_1_up_a2 = F.upsample(x5_dem_1, size=x2.size()[2:], mode='bilinear')     # bs,64,56,56  ***
        x5_dem_1_up_a1= F.upsample(x5_dem_1, size=x1.size()[2:], mode='bilinear')     # bs,64,56,56  ***


        x5_dem_1_up_map1 = self.conv_3(x5_dem_1_up)
        x4_dem_1_map1 = self.conv_3(x4_dem_1)
        x5_dem_1_up_map2 = self.conv_5(x5_dem_1_up)
        x4_dem_1_map2 = self.conv_5(x4_dem_1)
        x5_4 = self.x5_x4(
            abs(x5_dem_1_up - x4_dem_1)+abs(x5_dem_1_up_map1-x4_dem_1_map1)+abs(x5_dem_1_up_map2-x4_dem_1_map2))


        x4_dem_1_up = F.upsample(x4_dem_1, size=x3.size()[2:], mode='bilinear')

        x4_dem_1_up_a2 = F.upsample(x4_dem_1, size=x2.size()[2:], mode='bilinear')  #****
        x4_dem_1_up_a1 = F.upsample(x4_dem_1, size=x1.size()[2:], mode='bilinear')  #****


        x4_dem_1_up_map1 = self.conv_3(x4_dem_1_up)
        x3_dem_1_map1 = self.conv_3(x3_dem_1)
        x4_dem_1_up_map2 = self.conv_5(x4_dem_1_up)
        x3_dem_1_map2 = self.conv_5(x3_dem_1)

        x5_dem_1_up_a3_map1 = self.conv_3(x5_dem_1_up_a3)  #****
        x5_dem_1_up_a3_map2 = self.conv_5(x5_dem_1_up_a3)  #****

        x4_3 = self.x4_x3(
            abs(x4_dem_1_up - x3_dem_1)+abs(x4_dem_1_up_map1-x3_dem_1_map1)+abs(x4_dem_1_up_map2-x3_dem_1_map2)
             + abs(x5_dem_1_up_a3 - x3_dem_1)+abs(x5_dem_1_up_a3_map1-x3_dem_1_map1)+abs(x5_dem_1_up_a3_map2-x3_dem_1_map2)
              )

        x3_dem_1_up = F.upsample(x3_dem_1, size=x2.size()[2:], mode='bilinear')

        x3_dem_1_up_a1 = F.upsample(x3_dem_1, size=x1.size()[2:], mode='bilinear')

        x3_dem_1_up_map1 = self.conv_3(x3_dem_1_up)
        x2_dem_1_map1 = self.conv_3(x2_dem_1)
        x3_dem_1_up_map2 = self.conv_5(x3_dem_1_up)
        x2_dem_1_map2 = self.conv_5(x2_dem_1)

        x5_dem_1_up_a2_map1 = self.conv_3(x5_dem_1_up_a2)  #****
        x5_dem_1_up_a2_map2 = self.conv_5(x5_dem_1_up_a2)  #****

        x4_dem_1_up_a2_map1 = self.conv_3(x4_dem_1_up_a2)  #****
        x4_dem_1_up_a2_map2 = self.conv_5(x4_dem_1_up_a2)  #****

        x3_2 = self.x3_x2(
            abs(x3_dem_1_up - x2_dem_1)+abs(x3_dem_1_up_map1-x2_dem_1_map1)+abs(x3_dem_1_up_map2-x2_dem_1_map2)
             + abs(x5_dem_1_up_a2 - x2_dem_1)+abs(x5_dem_1_up_a2_map1-x2_dem_1_map1)+abs(x5_dem_1_up_a2_map2-x2_dem_1_map2)
              + abs(x4_dem_1_up_a2 - x2_dem_1)+abs(x4_dem_1_up_a2_map1-x2_dem_1_map1)+abs(x4_dem_1_up_a2_map2-x2_dem_1_map2)
              )
        



        x2_dem_1_up = F.upsample(x2_dem_1, size=x1.size()[2:], mode='bilinear')
        x2_dem_1_up_map1 = self.conv_3(x2_dem_1_up)
        x1_map1 = self.conv_3(x1_dem_1)
        x2_dem_1_up_map2 = self.conv_5(x2_dem_1_up)
        x1_map2 = self.conv_5(x1_dem_1)

        x5_dem_1_up_a1_map1 = self.conv_3(x5_dem_1_up_a1)  #****
        x5_dem_1_up_a1_map2 = self.conv_5(x5_dem_1_up_a1)  #****

        x4_dem_1_up_a1_map1 = self.conv_3(x4_dem_1_up_a1)  #****
        x4_dem_1_up_a1_map2 = self.conv_5(x4_dem_1_up_a1)  #****        

        x3_dem_1_up_a1_map1 = self.conv_3(x3_dem_1_up_a1)  #****
        x3_dem_1_up_a1_map2 = self.conv_5(x3_dem_1_up_a1)  #****  

        x2_1 = self.x2_x1(
            abs(x2_dem_1_up - x1_dem_1)+abs(x2_dem_1_up_map1-x1_map1)+abs(x2_dem_1_up_map2-x1_map2) 
            + abs(x5_dem_1_up_a1 - x1_dem_1)+abs(x5_dem_1_up_a1_map1-x1_map1)+abs(x5_dem_1_up_a1_map2-x1_map2)   
            + abs(x3_dem_1_up_a1 - x1_dem_1)+abs(x3_dem_1_up_a1_map1-x1_map1)+abs(x3_dem_1_up_a1_map2-x1_map2)       
            + abs(x4_dem_1_up_a1 - x1_dem_1)+abs(x4_dem_1_up_a1_map1-x1_map1)+abs(x4_dem_1_up_a1_map2-x1_map2)             
             )


        x5_4_up = F.upsample(x5_4, size=x4_3.size()[2:], mode='bilinear')
        x5_4_up_b2 = F.upsample(x5_4, size=x3_2.size()[2:], mode='bilinear')
        x5_4_up_b1 = F.upsample(x5_4, size=x2_1.size()[2:], mode='bilinear')


        x5_4_up_map1 = self.conv_3(x5_4_up)
        x4_3_map1 = self.conv_3(x4_3)
        x5_4_up_map2 = self.conv_5(x5_4_up)
        x4_3_map2 = self.conv_5(x4_3)
        x5_4_3 = self.x5_x4_x3(abs(x5_4_up - x4_3) +abs(x5_4_up_map1-x4_3_map1)+abs(x5_4_up_map2-x4_3_map2))


        x4_3_up = F.upsample(x4_3, size=x3_2.size()[2:], mode='bilinear')
        x4_3_up_map1 = self.conv_3(x4_3_up)
        x3_2_map1 = self.conv_3(x3_2)
        x4_3_up_map2 = self.conv_5(x4_3_up)
        x3_2_map2 = self.conv_5(x3_2)

        x5_dem_1_up_b2_map1 = self.conv_3(x5_4_up_b2)  #****
        x5_dem_1_up_b2_map2 = self.conv_5(x5_4_up_b2)  #****

        x4_3_2 = self.x4_x3_x2(abs(x4_3_up - x3_2)+abs(x4_3_up_map1-x3_2_map1)+abs(x4_3_up_map2-x3_2_map2) 
                               + abs(x5_4_up_b2 - x3_2)+abs(x5_dem_1_up_b2_map1-x3_2_map1)+abs(x5_dem_1_up_b2_map2-x3_2_map2) 
                               )

        x4_3_up_a1 = F.upsample(x4_3, size=x2_1.size()[2:], mode='bilinear')
        x3_2_up = F.upsample(x3_2, size=x2_1.size()[2:], mode='bilinear')
        x3_2_up_map1 = self.conv_3(x3_2_up)
        x2_1_map1 = self.conv_3(x2_1)
        x3_2_up_map2 = self.conv_5(x3_2_up)
        x2_1_map2 = self.conv_5(x2_1)

        x5_dem_1_up_b1_map1 = self.conv_3(x5_4_up_b1)  #****
        x5_dem_1_up_b1_map2 = self.conv_5(x5_4_up_b1)  #****

        x4_3_up_a1_map1 = self.conv_3(x4_3_up_a1)  #****
        x4_3_up_a1_map2 = self.conv_5(x4_3_up_a1)  #****

        x3_2_1 = self.x3_x2_x1(abs(x3_2_up - x2_1)+abs(x3_2_up_map1-x2_1_map1)+abs(x3_2_up_map2-x2_1_map2)
                               + abs(x5_4_up_b1 - x2_1)+abs(x5_dem_1_up_b1_map1-x2_1_map1)+abs(x5_dem_1_up_b1_map2-x2_1_map2)
                               + abs(x4_3_up_a1 - x2_1)+abs(x4_3_up_a1_map1-x2_1_map1)+abs(x4_3_up_a1_map2-x2_1_map2)
                                )


        x5_4_3_up = F.upsample(x5_4_3, size=x4_3_2.size()[2:], mode='bilinear')
        x5_4_3_up_c1 = F.upsample(x5_4_3, size=x3_2_1.size()[2:], mode='bilinear')


        x5_4_3_up_map1 = self.conv_3(x5_4_3_up)
        x4_3_2_map1 = self.conv_3(x4_3_2)
        x5_4_3_up_map2 = self.conv_5(x5_4_3_up)
        x4_3_2_map2 = self.conv_5(x4_3_2)

        x5_dem_1_up_c1_map1 = self.conv_3(x5_4_3_up_c1)  #****
        x5_dem_1_up_c1_map2 = self.conv_5(x5_4_3_up_c1)  #****

        x5_4_3_2 = self.x5_x4_x3_x2(
            abs(x5_4_3_up - x4_3_2)+abs(x5_4_3_up_map1-x4_3_2_map1)+abs(x5_4_3_up_map2-x4_3_2_map2)
              )


        x4_3_2_up = F.upsample(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear')
        x4_3_2_up_map1 = self.conv_3(x4_3_2_up)
        x3_2_1_map1 = self.conv_3(x3_2_1)
        x4_3_2_up_map2 = self.conv_5(x4_3_2_up)
        x3_2_1_map2 = self.conv_5(x3_2_1)
        x4_3_2_1 = self.x4_x3_x2_x1(
            abs(x4_3_2_up - x3_2_1) +abs(x4_3_2_up_map1-x3_2_1_map1)+abs(x4_3_2_up_map2-x3_2_1_map2)
             + abs(x5_4_3_up_c1 - x3_2_1)+abs(x5_dem_1_up_c1_map1-x3_2_1_map1)+abs(x5_dem_1_up_c1_map2-x3_2_1_map2))

        x5_dem_4 = self.x5_dem_4(x5_4_3_2)
        x5_dem_4_up = F.upsample(x5_dem_4, size=x4_3_2_1.size()[2:], mode='bilinear')
        x5_dem_4_up_map1 = self.conv_3(x5_dem_4_up)
        x4_3_2_1_map1 = self.conv_3(x4_3_2_1)
        x5_dem_4_up_map2 = self.conv_5(x5_dem_4_up)
        x4_3_2_1_map2 = self.conv_5(x4_3_2_1)
        x5_4_3_2_1 = self.x5_x4_x3_x2_x1(
            abs(x5_dem_4_up - x4_3_2_1)+abs(x5_dem_4_up_map1-x4_3_2_1_map1)+abs(x5_dem_4_up_map2-x4_3_2_1_map2) )

        level4 = self.level4(x4_dem_1 + x5_4)                                   # 2,64,14,14
        level3 = self.level3(x3_dem_1 + x4_3 + x5_4_3)                          # 2,64,28,28
        level2 = self.level2(x2_dem_1 + x3_2 + x4_3_2 + x5_4_3_2)               # 2,64,56,56
        level1 = self.level1(x1_dem_1 + x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1)  # 2,64,56,56



        # output5 = self.x5_dem_5(x5)  # 2,2048,7,7  ->  2,64,7,7
        # output4 = self.output4(self.transposed_conv(output5) + level4)  # 2,64,14,14
        # output3 = self.output3(self.transposed_conv(output4) + level3)  # 2,64,28,28
        # output2 = self.output2(self.transposed_conv(output3) + level2)  # 2,64,56,56
        # # attention
        # atten = self.atten1(F.upsample(output2, size=level1.size()[2:], mode='bilinear'))   # 2,2,112,112
        # atten1 = torch.softmax(atten, dim=1)  # 2,2,112,112
        # atten2 = atten1[:, 1:2, :, :]         # 2,1,112,112
        # output1 = self.output1(F.upsample(output2,size=level1.size()[2:], mode='bilinear') + level1)  # 2,4,112,112
        # output1 = output1 * atten2
        # output = F.upsample(output1, size=input.size()[2:], mode='bilinear')



        output5 = self.x5_dem_5(x5)                                                     # 2,2048,7,7  ->  2,64,7,7
        output4 = self.output4(F.upsample(output5,size=level4.size()[2:], mode='bilinear') + level4)  # 2,64,14,14
        output3 = self.output3(F.upsample(output4,size=level3.size()[2:], mode='bilinear') + level3)  # 2,64,28,28
        output2 = self.output2(F.upsample(output3, size=level2.size()[2:], mode='bilinear') + level2)
        # attention
        atten = self.atten1(F.upsample(output2, size=level1.size()[2:], mode='bilinear'))   # 2,2,56,56
        atten1 = torch.softmax(atten, dim=1)  # 2,1,28,28 
        atten2 = atten1[:, 1:2, :, :]
        output1 = self.output1(F.upsample(output2,size=level1.size()[2:], mode='bilinear') + level1)  # 2,64,56,56
        output1 = output1 * atten2
        # output2 = self.output2(F.upsample(output3,size=level2.size()[2:], mode='bilinear') + level2)  # 2,64,56,56
        
        output = F.upsample(output1, size=input.size()[2:], mode='bilinear')

        mu = mu1 + mu2 + mu3 + mu4 + mu5

        if self.training:
            return output, mu, atten1
        return output, atten1
    
class LossNet(torch.nn.Module):
    def __init__(self, resize=False):
        super(LossNet, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights="IMAGENET1K_V1").features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights="IMAGENET1K_V1").features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights="IMAGENET1K_V1").features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights="IMAGENET1K_V1").features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):  # 24,352,352   24,352,352
        if input.shape[1] != 3:
            input = input.unsqueeze(1).expand(input.shape[0], 3, 224, 224)
            target = target.unsqueeze(1).expand(input.shape[0], 3, 224, 224)  # 1,3,352,352
        input = (input-self.mean) / self.std   # 1,3,1,1   1,3,1,1  1,3,352,352
        target = (target-self.mean) / self.std  
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)   # 1,3,224,224
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False) # 1,3,224,224
        loss = 0.0
        x = input
        y = target

        for block in self.blocks:
            x = block(x)  # 1,64,224,224
            y = block(y)  # 1,64,224,224
            loss += torch.nn.functional.mse_loss(x, y)
        return loss

if __name__ == '__main__':
    model = M2SNet().cuda()
    x = torch.rand(2, 3, 224,224).cuda()
    output4, mu, atten1 = model(x)
    print(output4.shape)  # 2,2,224,224
    print(mu.shape)  # 2,2,224,224
    print(atten1.shape)  # 2,2,224,224


# from thop import profile
# # https://blog.csdn.net/haima1998/article/details/121365118
# if __name__ == '__main__':
#     net = M2SNet().cuda()
#     input = torch.randn(1,3,160,288).cuda()# 1, 3, 224, 224
#     flops, params = profile(net,inputs=(input, ))

#     print('Total FLOPs of parameters : %f G' % (flops/1000000000))
#     print('Total number of parameters : %f M' % (params/1000000))

