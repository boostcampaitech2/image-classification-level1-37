import torch
import pandas as pd
from PIL import Image

from glob import glob
import os
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from train import train

class TestDataset(Dataset):

    def __init__(self, transform):
        self.path = '/opt/ml/input/data/eval/info.csv'
        self.transform = transform
        tmp_data = pd.read_csv(self.path)
        self.validation_image_path = os.path.split(self.path)[0]
        self.validation_image_path = os.path.join(self.validation_image_path, "images")
        self.X = tmp_data.ImageID
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        X = Image.open(os.path.join(self.validation_image_path,X))
        if self.transform:
            X = self.transform(X)
        return torch.Tensor(X)