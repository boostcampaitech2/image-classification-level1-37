import torchvision
import torch
from tqdm import tqdm, notebook
from torch.utils.data import DataLoader
from testdataset import TestDataset

def eval(myModels, device):

    transform = torchvision.transforms.ToTensor()
    dataset = TestDataset(transform)
    loader = DataLoader(dataset, shuffle=False)
    total_label = []
    for myModel in myModels:
        myModel.eval()
        label = []
        for images in tqdm(loader):
          images = images.to(device)
          with torch.set_grad_enabled(False): 
            logits = myModel(images)
            _, preds = torch.max(logits, 1)
            label.append(preds.tolist()[0])
        total_label.append(label)
    class_sum = list(map(lambda x,y,z: 6*x+3*y+z, total_label[0],total_label[1],total_label[2]))
    return class_sum