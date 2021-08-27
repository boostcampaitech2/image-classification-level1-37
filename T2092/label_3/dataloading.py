import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
torch.manual_seed(37)
from labeling import CustomDataset
from data_dir import cfg_train



class dataloaders:

    def __init__(self):
        train_transform_vanilla = A.Compose([
            A.Normalize(mean=(0.560,0.524,0.501), std=(0.233,0.243,0.246), max_pixel_value=255.0, always_apply=False, p=1.0),
            ToTensorV2()
        ])

        BATCH_SIZE = 64
        mask_train = CustomDataset(cfg_train.img_dir, train = True, transform = train_transform_vanilla, data_name='mask')
        gender_train = CustomDataset(cfg_train.img_dir, train = True, transform = train_transform_vanilla, data_name='gender')
        age_train = CustomDataset(cfg_train.img_dir, train = True, transform = train_transform_vanilla, data_name='age')
        val_size = len(mask_train)//5
        train_size = len(mask_train)-val_size

        mask_train_dataset, mask_val_dataset = torch.utils.data.random_split(mask_train, [train_size,val_size])
        gender_train_dataset, gender_val_dataset = torch.utils.data.random_split(gender_train, [train_size,val_size])
        age_train_dataset, age_val_dataset = torch.utils.data.random_split(age_train, [train_size, val_size])

        mask_train_dl = torch.utils.data.DataLoader(mask_train_dataset, batch_size = BATCH_SIZE, shuffle = True)
        mask_val_dl = torch.utils.data.DataLoader(mask_val_dataset, batch_size = BATCH_SIZE, shuffle = True)
        gender_train_dl = torch.utils.data.DataLoader(gender_train_dataset, batch_size = BATCH_SIZE, shuffle = True)
        gender_val_dl = torch.utils.data.DataLoader(gender_val_dataset, batch_size = BATCH_SIZE, shuffle = True)
        age_train_dl = torch.utils.data.DataLoader(age_train_dataset, batch_size = BATCH_SIZE, shuffle = True)
        age_val_dl = torch.utils.data.DataLoader(age_val_dataset, batch_size = BATCH_SIZE, shuffle = True)

        mask_data_dict = {
            "train": mask_train_dl,
            "val": mask_val_dl
        }
        gender_data_dict = {
            "train": gender_train_dl,
            "val": gender_val_dl
        }
        age_data_dict = {
            "train": age_train_dl,
            "val": age_val_dl
        }

        data_dict = {
            "mask": mask_data_dict,
            "gender": gender_data_dict,
            "age": age_data_dict,
        }
        self.data_dict = data_dict

    def get_dict(self):
        return self.data_dict
