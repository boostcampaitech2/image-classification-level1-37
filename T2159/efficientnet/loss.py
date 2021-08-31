
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import torch.nn as nn
import numpy as np
device = torch.device('cuda')

def weighted_cross_entropy_loss(dataset):
    labels = [0 for _ in range(18)]
    for idx in tqdm(range(len(dataset))):
        _,y = dataset[idx]
        labels[y] += 1.0

    label = torch.tensor(labels)
    label = label.unsqueeze(0)
    label = label/torch.sum(label)
    weights = 1.0 / label
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights,reduction='sum').to(device)
    return loss_fn
"""or
label = torch.tensor(labels)
label = label.unsqueeze(0)
label = torch.sum(label) / label
label"""