import torchvision
import torch
import numpy as np
import os, sys
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, SubsetRandomSampler
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import random
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import StratifiedKFold


import numpy as np
from torch.utils.data import Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
sys.path.append(os.path.abspath('..'))


def seed_everything(seed):
    """
    동일한 조건으로 학습을 할 때, 동일한 결과를 얻기 위해 seed를 고정시킵니다.
    
    Args:
        seed: seed 정수값
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
seed_everything(42)

class maskImageDataset(Dataset):
    def __init__(self,img_path,label_path,transform=True):
        self.image = self.load_image(img_path)
        self.transform = transform
        self.label_path = label_path
        self.label = self.load_label(label_path)


        
    def __getitem__(self,idx):
        image, label= Image.open(self.image[idx]), self.label[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    def __len__(self):
            return len(self.label)

    def load_image(self,paths):
        img_lst = []
        for dic in os.listdir(paths):
            if '._' in dic or 'ipynb_checkpoints' in dic:
                continue
            dir_path = paths + '/'+ dic
            for image in os.listdir(dir_path):
                if '._' in image or 'ipynb_checkpoints' in image:
                    continue
                image_path = dir_path + '/' + image 
                img_lst.append(image_path)
        return img_lst
    
    def load_label(self, paths):
        df = pd.read_csv(os.path.join(paths, "train_label_fix_1.csv"))
        return df['label']


img_path = '/opt/ml/input/data/train/images'
label_path = '/opt/ml/input/data/train'
transform = transforms.Compose([
    transforms.Resize((512,384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.55800916, 0.51224077, 0.47767341), std=(0.21817792, 0.23804603, 0.25183411))
]) 
DATA = maskImageDataset(img_path,label_path,transform)


# ImageNet에서 학습된 ResNet 18 딥러닝 모델을 불러옴
imagenet_resnet18 = torchvision.models.resnet18(pretrained=True)
print("네트워크 필요 입력 채널 개수", imagenet_resnet18.conv1.weight.shape[1])
print("네트워크 출력 채널 개수 (예측 class type 개수)", imagenet_resnet18.fc.weight.shape[0])
# -- parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 학습 때 GPU 사용여부 결정. Colab에서는 "런타임"->"런타임 유형 변경"에서 "GPU"를 선택할 수 있음
batch_size = 16
num_workers = 4
num_classes = 3

num_epochs = 5  # 학습할 epoch의 수
lr = 1e-3
lr_decay_step = 10
criterion_name = 'cross_entropy' # loss의 이름

train_log_interval = 20  # logging할 iteration의 주기
name = "resnet18_model_results"  # 결과를 저장하는 폴더의 이름

train_dataloader = torch.utils.data.DataLoader(DATA, batch_size=batch_size, shuffle=True, num_workers=2)
import math
target_model = imagenet_resnet18
INPUT_NUM = 3
CLASS_NUM = 18
# for param in target_model.parameters():
#     param.requires_grad = False
# target model의 입력 크기와 출력 크기를 변경하여 줍니다 :) 새로운 네트워크 가중치를 만들어서 기존 부분 중 일부를 변경합니다.
target_model.conv1 = torch.nn.Conv2d(INPUT_NUM, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
target_model.fc = torch.nn.Linear(in_features=512, out_features = CLASS_NUM, bias=True)

# 새롭게 넣은 네트워크 가중치를 xavier uniform으로 초기화 해줍니다.
# (참고.해보기) 왜 xavier uniform으로 초기화해줄까요? - 관련 논문을 읽고 (https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) 생각해봅시다
torch.nn.init.xavier_uniform_(target_model.fc.weight)
stdv = 1. / math.sqrt(target_model.fc.weight.size(1))
target_model.fc.bias.data.uniform_(-stdv, stdv)

print("네트워크 필요 입력 채널 개수", target_model.conv1.weight.shape[1])
print("네트워크 출력 채널 개수 (예측 class type 개수)", target_model.fc.weight.shape[0])

def getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers):
    # 인자로 전달받은 dataset에서 train_idx에 해당하는 Subset 추출
    train_set = torch.utils.data.Subset(dataset,
                                        indices=train_idx)
    # 인자로 전달받은 dataset에서 valid_idx에 해당하는 Subset 추출
    val_set   = torch.utils.data.Subset(dataset,
                                        indices=valid_idx)
    
    # 추출된 Train Subset으로 DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True
    )
    # 추출된 Valid Subset으로 DataLoader 생성
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False
    )
    
    # 생성한 DataLoader 반환
    return train_loader, val_loader

    os.makedirs(os.path.join(os.getcwd(), 'results', name), exist_ok=True)

# 5-fold Stratified KFold 5개의 fold를 형성하고 5번 Cross Validation을 진행합니다.
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits)

counter = 0
patience = 30
accumulation_steps = 2
best_val_acc = 0
best_val_loss = np.inf
best_f1 = 0

# Stratified KFold를 사용해 Train, Valid fold의 Index를 생성합니다.
# labels 변수에 담긴 클래스를 기준으로 Stratify를 진행합니다. 
for i, (train_idx, valid_idx) in enumerate(skf.split(DATA.image, DATA.label)):
    
    # 생성한 Train, Valid Index를 getDataloader 함수에 전달해 train/valid DataLoader를 생성합니다.
    # 생성한 train, valid DataLoader로 이전과 같이 모델 학습을 진행합니다. 
    train_loader, val_loader = getDataloader(DATA, train_idx, valid_idx, batch_size, num_workers)

    # -- model
    model = target_model
    if torch.cuda.is_available():
        model.cuda()

    # -- loss & metric
    criterion = torch.nn.CrossEntropyLoss() # 분류 학습 때 많이 사용되는 Cross entropy loss를 objective function으로 사용 - https://en.wikipedia.org/wiki/Cross_entropy
    optimizer = torch.optim.Adam(target_model.parameters(), lr=lr) # weight 업데이트를 위한 optimizer를 Adam으로 사용함

    scheduler = StepLR(optimizer, lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=f"results/cv{i}_{name}")
    for epoch in range(num_epochs):

        epoch_f1 = 0
        n_iter = 0
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            
             # -- Gradient Accumulation
            if (idx+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % train_log_interval == 0:
                train_loss = loss_value / train_log_interval
                train_acc = matches / batch_size / train_log_interval
                current_lr = scheduler.get_last_lr()
                print(
                    f"Epoch[{epoch}/{num_epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )

                loss_value = 0
                matches = 0
            epoch_f1 += f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='macro')
            n_iter += 1

        epoch_f1 = epoch_f1/n_iter
        print(f"f1 score: {epoch_f1:.4f}")
        if epoch_f1 > best_f1:
            torch.save(model, f"/opt/ml/code/results/{name}/{epoch:03}_f1_{epoch_f1:4.2%}.pt")
            best_f1 = epoch_f1
            print("New best model for f1 score! saving the model..")
            counter = 0
        else:
            counter +=1
        if counter > patience:
            print("Early Stopping...")
            break

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(valid_idx)

            # Callback1: validation accuracy가 향상될수록 모델을 저장합니다.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc > best_val_acc:
                #torch.save(model, f"/opt/ml/code/results/{name}/{epoch:03}_accuracy_{val_acc:4.2%}.pt")
                best_val_acc = val_acc
                #counter = 0
            else:
                #counter += 1
            # Callback2: patience 횟수 동안 성능 향상이 없을 경우 학습을 종료시킵니다.
            if counter > patience:
                print("Early Stopping...")
                break


            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )