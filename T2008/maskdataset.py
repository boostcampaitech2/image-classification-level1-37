import os
import pandas as pd
import platform
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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
        self.path = {"train_label" : '/opt/ml/input/data/train/train_with_labels_fix_wrong_data.csv',
                     "train_vanilla" : '/opt/ml/input/data/train/train.csv',
                     "validation" : '/opt/ml/input/data/eval/info.csv'}
        if not platform.system() in ["Windows", "Linux"]:
                self.path = {k:v.replace("/","\\") for k,v in self.path.items()}
        self.transform = transform
        # self.transform=transforms.Compose([
        #     # transforms.Resize((512,384), Image.BILINEAR),
        #     transforms.ToTensor(),
        #     # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        #     transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.2,0.2,0.2))
        #     ])
        self.train = train
        #Train dataset
        if self.train: #이미 labeling data가 포함된 csv 파일이 있는 경우
            self.data = pd.read_csv(self.path["train_label"])
            self.X = self.data.img_path.map(Image.open)
            self.y = self.data["label"]
        else: #Test or Validation Dataset
            self.data = pd.read_csv(self.path["validation"])
            image_path = os.path.split(self.path["validation"])[0]
            image_path = os.path.join(image_path, "images")
            self.X = self.data.ImageID.map(lambda x : Image.open(os.path.join(image_path, x)))

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        X = self.transform(X)
        
        if self.train:
            y = self.y[idx]
            return torch.tensor(X), torch.tensor(y)
        else:
            return torch.tensor(X)
