import os
import pandas as pd
from PIL import Image
import re
from collections import OrderedDict
from timm.models import efficientnet
from tqdm import tqdm, notebook
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
from loss import weighted_cross_entropy_loss
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
loss_function = weighted_cross_entropy_loss(DATA)

efficientnet = timm.create_model('tf_efficientnet_b3', pretrained=True, num_classes = 18)
# efficientnet.classifier = nn.Sequential(
#     nn.Dropout(0.25),
#     nn.Linear(2048, 18)
#     )
#efficientnet = torch.load('/opt/ml/model/efficientnetv2_l/last_f1_score_10epoch_0.7471227911450713.pt')
if torch.cuda.is_available():
    efficientnet.cuda()

losses, acc, f1score = training(efficientnet,x,loaders,device,loss_function,20)
eval(efficientnet,'tf_efficient_b3_epoch20.csv')
