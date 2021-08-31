import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from PIL import Image
import numpy as np

def make_more_image(transform,text):
    text = 'a' + text #augmetntatoin한 이미지임을 표기하기위해 a를 이미지의 이름 앞에 붙였습니다.
    paths = '/opt/ml/input/data/train/images'
    x = []
    y = []
    for dic in tqdm(os.listdir(paths)):
        if '._' in dic or 'ipynb_checkpoints' in dic:
            continue
        dir_path = paths + '/'+ dic
        age = int(dic[-2:])
        if age>= 60: #augmentaion하려는 조건을 붙이면 됩니다.
            for image in os.listdir(dir_path):
                if '._' in image or 'ipynb_checkpoints' in image:
                    continue
                if image[0] != 'a':
                    images= text + image
                    image_path = os.path.join(dir_path,image)
                    image = np.asarray(Image.open(image_path))
                    newimage = transform(image=image)['image']
                    img = Image.fromarray(newimage, 'RGB')
                    newimgpath = os.path.join(dir_path,images) 
                    img.save(newimgpath, format=None)
def del_aug(path): #  augmentaion한 이미지를 삭제하는 함수
    paths = '/opt/ml/input/data/train/images'
    for dic in tqdm(os.listdir(paths)):
        if '._' in dic or 'ipynb_checkpoints' in dic:
            continue
        dir_path = paths + '/'+ dic
        age = int(dic[-2:])
        if age >= 60: #삭제할 조건
            for image in os.listdir(dir_path):
                if image[0] == 'a':
                    image_path = dir_path + '/' + image
                    os.remove(image_path)
""" 
예시
train_transform_RGB = A.Compose([
    A.HorizontalFlip(always_apply=True),
])
del_aug('/opt/ml/input/data/train/images')
make_more_image(train_transform_flip,'ToGray')
del_aug('/opt/ml/input/data/train/images')
"""