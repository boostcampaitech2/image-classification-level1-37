import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

efficientnet = timm.create_model('tf_efficientnet_b0', pretrained=True,num_classes=18)
for param in efficientnet.parameters():
    param.requires_grad = False
efficientnet.conv_head = nn.Sequential(nn.Conv2d(320, 640, kernel_size=(3, 3), stride=(1, 1), bias=False),
                            nn.BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                            nn.SiLU(inplace=True),nn.Dropout(p=0.3),nn.Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False))    
efficientnet.classifier = nn.Linear(in_features=1280, out_features=18, bias=True)

if torch.cuda.is_available():
    efficientnet.cuda()
