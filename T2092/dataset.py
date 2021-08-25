
import torchvision
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from glob import glob
import os
from torch.utils.data import Dataset, DataLoader
%matplotlib inline



class cfg_train:
    data_dir = '/opt/ml/input/data/train'
    img_dir = 'images'
    df_path = 'train.csv'
class cfg_test:
    data_dir = '/opt/ml/input/data/eval'
    img_dir = 'images'
    df_path = 'info.csv'

class CustomDataset(Dataset):
    def __init__(self, path, df_path, train=True, transform = None):
        self.path = path
        self.transform = transform
        self.train = train
        if self.train == True:
            self.df = pd.read_csv(os.path.join(self.path, df_path)).drop(['id','race'], axis=1)
            self.DF = self._getlabel(os.path.join(self.path, 'images'), self.df.values)
            self.X = self.DF['ImageID'].values
            self.y = self.DF['ans'].values
            self.DF.to_csv(path_or_buf = os.path.join(self.path,"labeling.csv"), index=False)
        else:
            self.df = pd.read_csv(os.path.join(self.path, df_path)).drop('ans', axis=1)
            self.X = self.df.values.squeeze()
            self.X = np.array(list(map(lambda x: os.path.join(self.path, cfg_test.img_dir, x), self.X)))


    def __len__(self):
        len_dataset = len(self.X)
        return len_dataset

    def __getitem__(self, idx):
        img = Image.open(self.X[idx])
        if self.transform:
            img = self.transform(img)
        if self.train == True:
            y = self.y[idx]
            return img, y
        else:
            return img

    def _getlabel(self, img_dir, data):
        self.direc=[]
        y=[]
        for row in tqdm(data):
            for path in glob(os.path.join(img_dir,row[2],'*')):
                self.direc.append(np.concatenate((row, np.reshape(path,1)),axis=0))
        mask_pd = pd.DataFrame(self.direc, columns = ['gender','age','person','path']).drop('person',axis=1)
        for gender, age, path in mask_pd.values:
            temp_class=0
            if 'incorrect' in path:
                temp_class+=6
            if 'normal' in path:
                temp_class+=12
            if gender == 'female':
                temp_class+=3
            if age >= 30 and age < 60:
                temp_class+=1
            if age >= 60:
                temp_class+=2
            y.append(temp_class)
        return pd.concat([mask_pd['path'], pd.Series(data=y,name = 'ans')],axis=1).rename(columns={'path':'ImageID'})
    
if __name__ == "__main__":
    mask_train = CustomDataset(cfg_train.data_dir,'train.csv', train = True, transform = torchvision.transforms.ToTensor())
