import torch
from tqdm import tqdm, notebook
import os
from sklearn.metrics import f1_score


def train(name,myModel, NUM_EPOCH,optimizer,loss_fn,dataloaders, data_dir, device):
    ckpt_dir = os.path.join(data_dir, 'checkpoints','model_3_1')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    best_test_accuracy = 0.
    best_test_loss = 999.
    best_test_f1score =  0.
    target_model = myModel
    from_update=0
    for epoch in range(NUM_EPOCH):
      for phase in ["train", "val"]:
        running_loss = 0.
        running_acc = 0.
        if phase == "train":
          target_model.train()
        elif phase == "val":
          target_model.eval()
        for ind, (images, labels) in enumerate(tqdm(dataloaders[phase])):
          images = images.to(device)
          labels = labels.to(device)

          optimizer.zero_grad()

          with torch.set_grad_enabled(phase == "train"):
            logits = target_model(images)
            _, preds = torch.max(logits, 1) 
            loss = loss_fn(logits, labels)

            if phase == "train":
              loss.backward() 
              optimizer.step()

          running_loss += loss.item() * images.size(0) 
          running_acc += torch.sum(preds == labels.data)
          if phase == "train":
            print(f"{name}\tloss: {running_loss/((ind+1)*64):.3f},\tacc: {running_acc/((ind+1)*64):.3f},\t{ind}/226")
          else:
            print(f"{name}\tloss: {running_loss/((ind+1)*64):.3f},\tacc: {running_acc/((ind+1)*64):.3f},\t{ind}/56")

        epoch_f1score = f1_score(labels.cpu(),preds.cpu(),average="macro")
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_acc / len(dataloaders[phase].dataset)

        print(f"현재 epoch-{epoch}의 {phase}-데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}, 평균 f1score: {epoch_f1score:.3f}")
        if phase == "val" and best_test_f1score < epoch_f1score:
          best_test_f1score = epoch_f1score
        if phase == "val" and best_test_accuracy < epoch_acc:
          from_update = 0
          best_test_accuracy = epoch_acc
        if phase == "val" and best_test_loss > epoch_loss:
          torch.save({
            'epoch': epoch,
            'model_state_dict': target_model.state_dict(),
            'model': target_model,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
          },os.path.join(data_dir, 'checkpoints','model_3_1', f"max_model_{name}_{epoch}_{epoch_loss:.3f}.pt"))
          best_test_loss = epoch_loss
      from_update+=1
      if from_update > 2:
        break
    print("학습 종료!")
    print(f"최고 accuracy : {best_test_accuracy}, 최고 낮은 loss : {best_test_loss}, 최고 높은 f1score: {best_test_f1score}")