import os
import numpy as np
import pandas as pd
from PIL import Image

import torch.utils.data as data
#import img_stat
from data_dir import cfg_train


df = pd.read_csv(os.path.join(cfg_train.data_dir, cfg_train.df_path))

class MaskLabels:
    mask = 0
    incorrect = 1
    normal = 2

class GenderLabels:
    male = 0
    female = 1

class AgeGroup:
    map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 60 else 2

class CustomDataset(data.Dataset):
    num_classes = 18

    _file_names = {
        "mask1.jpg": MaskLabels.mask,
        "mask2.jpg": MaskLabels.mask,
        "mask3.jpg": MaskLabels.mask,
        "mask4.jpg": MaskLabels.mask,
        "mask5.jpg": MaskLabels.mask,
        "incorrect_mask.jpg": MaskLabels.incorrect,
        "normal.jpg": MaskLabels.normal
    }

    def __init__(self, img_dir, train=False, transform=None, data_name='None'):
        self.image_paths = []
        self.mask_labels = []
        self.gender_labels = []
        self.age_labels = []
        self.img_dir = img_dir
        self.train = train
        self.data_name = data_name
        #self.mean, self.std = img_stat.get_img_stats(cfg_train.img_dir,df.path.values)
        self.transform = transform
        self.setup()

    def set_transform(self, transform):
        self.transform = transform
        
    def setup(self):
        #label 저장
        profiles = os.listdir(self.img_dir)
        for profile in profiles:
            for file_name, label in self._file_names.items():
                img_path = os.path.join(self.img_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.mask_labels.append(label)

                    id, gender, race, age = profile.split("_")
                    gender_label = getattr(GenderLabels, gender)
                    age_label = AgeGroup.map_label(age)

                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

    def __getitem__(self, index):
        # 이미지를 불러옵니다.
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        
        # 레이블을 불러옵니다.
        mask_label = self.mask_labels[index]
        gender_label = self.gender_labels[index]
        age_label = self.age_labels[index]
        labels = {
            'mask': mask_label,
            'gender': gender_label,
            'age': age_label
        }
        # multi_class_label = mask_label * 6 + gender_label * 3 + age_label
        
        # 이미지를 Augmentation 시킵니다.
        image_transform = self.transform(image=np.array(image))['image']
        if self.train == False:
            return image_transform
        return image_transform, labels[self.data_name]

    def __len__(self):
        return len(self.image_paths)

