import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from PIL import Image


class maskImageDataset(Dataset):
    def __init__(self,path,transform=None):
        self.image,self.label = self.labeling(path)
        self.transform = transform
        
    def __getitem__(self,idx):
        image,label = Image.open(self.image[idx]),self.label[idx]
        if self.transform:
            image = self.transform(image)
        return image,label

    def __len__(self):
            return len(self.label)

    def labeling(self,paths):
        x = []
        y = []
        for dic in os.listdir(paths):
            if '._' in dic or 'ipynb_checkpoints' in dic:
                continue
            dir_path = paths + '/'+ dic
            code = 0
            if dic[7] == 'f':
                code = 3
            age = int(dic[-2:])
            if age >= 60:
                code += 2
            elif age >=30:
                code += 1
            for image in os.listdir(dir_path):
                if '._' in image or 'ipynb_checkpoints' in image:
                    continue
                image_path = dir_path + '/' + image 
                x.append(image_path)
                label = [0 for _ in range(18)]
                y.append(self.age_labeling(image,code))
        return x,y
                
    def age_labeling(self,path,inputs):
        if 'incorrect_mask' in path:
            return inputs + 6
        elif 'normal' in path:
            return inputs + 12
        else:
            return inputs

""" 예시
        
train_dir = '/opt/ml/input/data/train/images'
transform = transforms.Compose([
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
]) 

DATA = maskImageDataset(train_dir,transform)
"""