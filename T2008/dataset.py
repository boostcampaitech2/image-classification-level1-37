import os
import pandas as pd
import platform
import torch
from PIL import Image
from labeling import MakeLabel
from torch.utils.data import Dataset

"""
    if __name__=="__main__":
    transform=transforms.Compose([transforms.Resize((512,384)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])

    trainDataSet = MaskDataset(path=path,
                            transform=transform,
                            train=True)

"""


class MaskDataset(Dataset):
    def __init__(self, transform, train=True):
        self.path = {"train_label" : '/opt/ml/input/data/train/train_with_labels.csv',
                     "train_vanilla" : '/opt/ml/input/data/train/train.csv',
                     "validation" : '/opt/ml/input/data/eval/info.csv'}
        if not platform.system() in ["Windows", "Linux"]:
                self.path = {k:v.replace("/","\\") for k,v in self.path.items()}
        self.transform = transform
        self.train = train
        
        #Train dataset
        if self.train: #이미 labeling data가 포함된 csv 파일이 있는 경우
            if os.path.exists(self.path["train_label"])==True:
                self.data = pd.read_csv(self.path["train_label"])
            else:
                data = pd.read_csv(self.path['train_vanilla'])
                image_path = os.path.split(self.path['train_vanilla'])[0]
                MakeLabel(data, image_path).labeling()
                self.data = pd.read_csv(self.path["train_label"])
            self.X = self.data.img_path.map(Image.open)
            self.y = self.data["label"]
        else: #Test or Validation Dataset
            tmp_data = pd.read_csv(self.path["validation"])
            validation_image_path = os.path.split(self.path["validation"])[0]
            validation_image_path = os.path.join(validation_image_path, "images")
            self.X = tmp_data.ImageID.map(lambda x : Image.open(os.path.join(validation_image_path, x)))

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        if self.transform:
            X = self.transform(X)
        
        if self.train:
            y = self.y[idx]
            return torch.Tensor(X), torch.Tensor(y)
        else:
            return torch.Tensor(X)


