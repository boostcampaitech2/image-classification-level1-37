import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

from sklearn.model_selection import StratifiedKFold


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import MaskBaseDataset,sj_MaskDataset
from loss import create_criterion

# 석진님 여기 label_fold 쪽에 주석 좀 달아주세요

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"
#
# def stratifiedkfold(image_dirs, label1):
#     stratified_kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
#     # Stratified K-Fold는 층화된 folds를 반환하는 기존 K-Fold의 변형된 방식. 각 집합에는 전체 집합과 거의 동일하게 클래스의 표본 비율이 포함된다. 불균형 클래스의 경우 사
#     # train에 2700개 중에서 4/5가 들어가고 valid에 1/5가 들어간다. 이걸 train할 때 다섯번 반복하면 된다.
#     fold_list = []
#
#     for train_data, valid_data in stratified_kfold.split(image_dirs, label1):  # split(x,y) x training data, y target
#
#         fold_list.append({'train': train_data, 'valid': valid_data})
#     return fold_list
#

def label_fold(image_dirs,seed): #
    stratified_kfold_label = []
    for image_dir in image_dirs:
        cnt = 0
        if 'female' in image_dir:
            cnt += 3
        else:
            cnt += 0

        age = int(image_dir.split('_')[3][:2])
        if age < 30:
            cnt += 0
        elif age < 58:
            cnt += 1
        else:
            cnt += 2
        stratified_kfold_label.append(cnt)
    stratified_kfold_label = np.array(stratified_kfold_label)
    stratified_kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    # Stratified K-Fold는 층화된 folds를 반환하는 기존 K-Fold의 변형된 방식이다. 각 집합에는 전체 집합과 거의 동일하게 클래스의 표본 비율이 포함된다.
    # 불균형 클래스의 경우 사용을 많이 한다. 이를 통해 데이터별로 가지는 클래스의 분포를 맞춰줄 수 있을 것으로 기대한다.
    # train에 2700개의 id 중에서 4/5가 들어가고 valid에 1/5가 들어간다. 이걸 다른 모양으로 train할 때 다섯번 반복하게 된다.
    fold_list = []
    for train_data, valid_data in stratified_kfold.split(image_dirs, stratified_kfold_label):
        fold_list.append({'train': train_data, 'valid': valid_data})
    return fold_list



def train(data_dir, model_dir, args):
    seed_everything(args.seed)
    path = Path('/opt/ml/input/data/train/images')
    image_dirs = [str(x) for x in list(path.glob('*')) if '._' not in str(x) and 'check' not in str(x)]
    image_dirs = np.array(image_dirs)
    fold_list = label_fold(image_dirs,args.seed) #
    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation

    num_classes = 18  # 18

    # -- augmentation
    train_transform_module = getattr(import_module("dataset"), args.augmentation)  # default: suckjin_augmentation
    train_transform = train_transform_module(
        width=args.resize[0],
        height =args.resize[1],
    )
    valid_transform_module = getattr(import_module("dataset"), args.valid_augmentation)  # default: suckjin_valid_augmentation
    valid_transform = valid_transform_module(
        width=args.resize[0],
        height =args.resize[1],
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)


    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: adma
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
#   scheduler = CosineAnnealingLR(optimizer,T_max=50)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)



    for fold in range(args.kfold):
        model.param_init()
        best_val_acc = 0
        best_val_loss = np.inf
        train_image_paths, valid_image_paths = [], []
        for train_dir in image_dirs[fold_list[fold - 1]['train']]:
            train_image_paths.extend(glob.glob(train_dir + '/*'))
        train_dataset = dataset_module(
            image_paths=train_image_paths,
        )
        for valid_dir in image_dirs[fold_list[fold - 1]['valid']]:
            valid_image_paths.extend(glob.glob(valid_dir + '/*'))
        valid_dataset = dataset_module(
            image_paths=valid_image_paths,
        )
        print(len(valid_dataset),len(valid_image_paths))
        train_dataset.set_transform(train_transform)
        valid_dataset.set_transform(valid_transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=True,
            pin_memory=use_cuda,
        )

        val_loader = DataLoader(
            valid_dataset,
            batch_size=args.valid_batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
        )
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                    loss_value = 0
                    matches = 0

            scheduler.step()
        # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                figure = None
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

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, train_dataset.mean, train_dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(valid_dataset)
                best_val_loss = min(best_val_loss, val_loss)
                if val_acc > best_val_acc:
                    print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best"+f"{fold}.pth")
                    best_val_acc = val_acc
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                print(
                    f"{fold}\n[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_figure("results", figure, epoch)
                print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)

    # Data and model checkpoints directoriesd
    parser.add_argument('--seed', type=int, default=37, help='random seed (default: 37)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--dataset', type=str, default='sj_MaskDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='sj_augmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--valid_augmentation', default='sj_valid_augmentation', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument("--resize", nargs="+", type=list, default=[96, 128], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--valid_batch_size', type=int, default=32, help='input batch size for validing (default: 1000)')
    parser.add_argument('--kfold', type=int, default=5, help=' kfold num')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=50, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    train(data_dir, model_dir, args)
