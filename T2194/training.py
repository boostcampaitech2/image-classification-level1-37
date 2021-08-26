import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm

def training(model,idx,loader,device,loss_fn,epochs):
    print(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    losses = [0]
    acc = [0]
    f1score = [0]
    model.train()
    for epoch in range(epochs):
        epoch_f1 = 0
        running_loss = 0
        running_acc = 0
        n_iter = 0
        for images,labels in tqdm(loader):
            images = images.to(device)
            output = model(images).to(device) 
            _, preds = torch.max(output, 1) # yhat
            labels = labels.to(device)
            
            optim.zero_grad() # parameter gradient를 업데이트 전 초기화함
            loss = loss_fn(output, labels)
            loss.backward()
            optim.step()
            
            running_loss += loss.item() # loss
            running_acc += torch.sum(preds == labels.data) # 정답 정확도
            epoch_f1 += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
            n_iter += 1
        epoch_f1 = epoch_f1/ n_iter
        running_loss = running_loss / idx
        running_acc = running_acc / idx
        losses.append(running_loss)
        acc.append(running_acc)
        f1score.append(epoch_f1)
        print(F'{epoch+1}epochs loss:{running_loss}  acc:{running_acc} f1_score :{epoch_f1},')
    return losses, acc, f1score