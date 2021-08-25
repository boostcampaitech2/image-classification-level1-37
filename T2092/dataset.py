import numpy as np
import pandas as pd
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from glob import glob
import os
from torch.utils.data import Dataset, DataLoader

class cfg:
    data_dir = '/opt/ml/input/data/train'
    img_dir = f'{data_dir}/images'
    df_path = f'{data_dir}/train.csv'

class CustomDataset(Dataset):
    def __init__(self, path, drop_features, train=True, transform = None):
        self.path = path
        self.transform = transform
        self.df = pd.read_csv(os.path.join(self.path, 'train.csv')).drop(drop_features, axis=1)
        self.DF = self._getlabel(os.path.join(self.path, 'images'), self.df.values)
        self.X = self.DF['ImageID'].values
        self.y = self.DF['ans'].values
        self.DF.to_csv(path_or_buf = os.path.join(cfg.data_dir,"labeling.csv"), index=False)

    def __len__(self):
        len_dataset = len(self.X)
        return len_dataset

    def __getitem__(self, idx):
        img = Image.open(self.X[idx])
        if self.transform:
            img = self.transform(img)
        y = self.y[idx]
        return img,y

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
    mask_data = CustomDataset(cfg.data_dir,["id",'race'],train=True)
