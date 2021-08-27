import os
import pandas as pd
from PIL import Image
import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import torch.nn as nn
import numpy as np

import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
import math
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import random
from loss import loss_fn
from maskImageDataset import maskImageDataset
from training import training,eval

device = torch.device('cuda')
train_dir = '/opt/ml/input/data/train/images'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

seed_everything(37) # random seed 37


transformss = A.Compose([
        #A.CenterCrop(350,200,p=1.0),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        ToTensorV2(),
])
device = torch.device('cuda')

DATA = maskImageDataset(train_dir,transformss) #데이터셋
x = len(DATA)

batch_size = 16
loaders = DataLoader(
    DATA,batch_size=batch_size,shuffle=True,num_workers=1)
loss_function = loss_fn(DATA)

efficientnet = timm.create_model('tf_efficientnet_b0', pretrained=True,num_classes=18)
#for param in efficientnet.parameters():
#    param.requires_grad = False unfreeze하니까 조금 더 잘 나왔어요

efficientnet.conv_head = nn.Sequential(nn.Conv2d(320, 640, kernel_size=(3, 3), stride=(1, 1), bias=False),
                            nn.BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                            nn.SiLU(inplace=True),nn.Dropout(p=0.3),nn.Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False))    
efficientnet.classifier = nn.Linear(in_features=1280, out_features=18, bias=True)

if torch.cuda.is_available():
    efficientnet.cuda()

loss,acc = training(efficientnet,x,loaders,device,loss_function,10)
eval(efficientnet,'efficeint.csv')
