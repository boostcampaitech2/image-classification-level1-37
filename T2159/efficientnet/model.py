import torchvision
import torch
import torch.nn as nn
import math
from collections import OrderedDict
import timm


torch.manual_seed(37)

class myefficientnet_b0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pretrain_model = timm.create_model('tf_efficientnet_b0', pretrained=True,num_classes=18)
        self.pretrain_model.conv_head = nn.Sequential(nn.Conv2d(320, 640, kernel_size=(3, 3), stride=(1, 1), bias=False),nn.BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                nn.SiLU(inplace=True),nn.Dropout(p=0.3),nn.Conv2d(640, 1280, kernel_size=(1, 1),
                stride=(1, 1), bias=False))    
    def forward(self,x):
        return self.pretrain_model(x)


