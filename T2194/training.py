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

def eval(model,text):
    test_dir = '/opt/ml/input/data/eval'
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    transform = transforms.Compose([
        Resize((512, 384), Image.BILINEAR),
        transforms.CenterCrop((350,280),p=1),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ])
    dataset = TestDataset(image_paths, transform)

    loader = DataLoader(
        dataset,
        shuffle=False
    )
    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
    device = torch.device('cuda')
    model.eval()

    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    all_predictions = []
    for images in tqdm(loader):
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join(test_dir, text), index=False)
    print('test inference is done!')