import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion
from sklearn.metrics import f1_score

import nni
import logging
from nni.utils import merge_parameter
logger_p = logging.getLogger('mask_AutoML')


""" 일반적인 training을 할 수 있는 파일입니다."""
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

def get_f1_score(y_true,y_pred):
    return f1_score(y_true.cpu(),y_pred.cpu(),average='macro')

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{args['model_dir']}/{args['name']}".
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

def train(args):
    seed_everything(args['seed'])
    save_dir = increment_path(os.path.join(args['model_dir'], args['name']))
    print("*"*30)
    print(f"{save_dir:*^30}")
    print("*"*30)

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args['dataset'])  # default: BaseAugmentation
    dataset = dataset_module(
        data_dir=args['data_dir'],
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args['augmentation'])  # default: BaseAugmentation
    transform = transform_module(
        height=args['resize_height'],
        width=args['resize_width'],
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args['batch_size'],
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args['valid_batch_size'],
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args['model'])  # default: BaseModel
    model = model_module(
        num_classes=num_classes,
        model_type=args['model_type'],
    ).to(device)
    model = torch.nn.DataParallel(model)
    
    # -- loss & metric
    #loss weight
    class_weights=None
    if args['weights_type'] != None: #string으로 None이 입력됨
        train_data_labels = train_set.dataset.output_labels
        weights_module = getattr(import_module("utils"), "GetClassWeights")
        weights = weights_module(train_data_labels)
        class_weights = weights.get_weights(args['weights_type'])
        class_weights = torch.FloatTensor(class_weights).to(device)

    criterion = create_criterion(args['criterion'], weight=class_weights,reduction=args['reduction'])
    
    if args['optimizer'].lower()=="adamp": #different optimizer package
        opt_module = getattr(import_module("adamp"), args['optimizer'])
    else:
        opt_module = getattr(import_module("torch.optim"), args['optimizer'])
    
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args['lr'],
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args['lr_decay_step'], gamma=0.1)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(args, f, ensure_ascii=False, indent=4)

    best_val_f1 = 0
    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(args['epochs']):
        # train loop
        model.train()
        f1_value = 0
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if args['cutmix']==True:
                rand_bbox_module = getattr(import_module("utils"), 'rand_bbox')
                rand_bbox = rand_bbox_module(inputs=inputs,
                                            labels=labels,
                                            beta1=args['beta1'],
                                            beta2=args['beta2'])
                inputs, lam, target_a, target_b = rand_bbox.get_cutmiximage_and_lam()

            optimizer.zero_grad()
            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            
            if args['cutmix']==True:
                loss = criterion(outs, target_a) * lam + criterion(outs, target_b) * (1. - lam)
            else:
                loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()
            f1_value += get_f1_score(labels,preds).item()
            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args['log_interval'] == 0:
                train_f1 = f1_value / args['log_interval']
                train_loss = loss_value / args['log_interval']
                train_acc = matches / args['batch_size'] / args['log_interval']
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch+1}/{args['epochs']}]({idx + 1}/{len(train_loader)}) || "
                    f"training f1 {train_f1:4.4} ||training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/f1", train_f1, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0
                f1_value = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            val_f1_items = []
            figure = None
            for idx, val_batch in enumerate(val_loader):
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                f1_item = get_f1_score(labels,preds).item()

                val_f1_items.append(f1_item)
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if (idx + 1) % 4 == 0:
                    val_f1_tmp = np.sum(val_f1_items) / (idx+1)
                    val_loss_tmp = np.sum(val_loss_items) / (idx+1)
                    val_acc_tmp = np.sum(val_acc_items) / args['valid_batch_size'] / (idx+1)
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch+1}/{args['epochs']}]({idx + 1}/{len(val_loader)}) || "
                        f"training f1 {val_f1_tmp:4.4} ||training loss {val_loss_tmp:4.4} || training accuracy {val_acc_tmp:4.2%} || lr {current_lr}"
                    )
                    
                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args['dataset'] != "MaskSplitByProfileDataset"
                    )
            
            val_f1 = np.sum(val_f1_items) / len(val_loader)
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)

            if val_f1 > best_val_f1:
                print(f"New best model for val f1 score : {val_f1:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_f1 = val_f1
                best_val_acc = val_acc
                best_val_loss = val_loss

            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            
            print(
                f"[Val] f1 : {val_f1:4.2%}, acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best f1 : {best_val_f1:4.2%}, best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/f1", val_f1, epoch)
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            # print()
            
            nni.report_intermediate_result(best_val_f1)
            logger_p.debug('test accuracy(val) is %g', val_f1)
            logger_p.debug('test accuracy(best) is %g', best_val_f1)
    
    nni.report_final_result(best_val_f1)
    logger_p.debug('Final result(f1 score) is %g', best_val_f1)
    logger_p.debug('Send final result done.')


def get_params():
    #argparse를 사용하기 위해 argumentparser 객체 생성
    parser = argparse.ArgumentParser(description='Mask')

    parser.add_argument('--seed', type=int, default=37, help='random seed (default: 37)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskBaseDataset)') #
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation_Max', help='data augmentation type (default: BaseAugmentation)') #CustomAugmentation
    parser.add_argument('--resize_height', type=int, default=512, help='input resize_height size for resize augmentation (default: 512)')# b0 224 ,b1 240 ,b2 260 ,b3 300 
    parser.add_argument('--resize_width', type=int, default=384, help='input resize_width size for resize augmentation (default: 384)') # b4 380 ,b5 456 ,b6 528 ,b7 600 
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 64)')
    parser.add_argument('--model', type=str, default='EffnetModel', help='model class name in model.py(EffnetModel, TimmModel)')
    parser.add_argument('--model_type', type=str, default='b4', help='model type (range: b0~b7 default b4)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--reduction', type=str, default='sum', help='criterion reduction type (default: sum)')
    parser.add_argument('--lr_decay_step', type=int, default=100, help='learning rate scheduler decay step (default: 100)')
    parser.add_argument('--log_interval', type=int, default=50, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--weights_type', type=str, default=None, help=' weights type (default: None)')
    parser.add_argument('--beta1', type=int, default=1, help='cutmix beta (default : 1)')
    parser.add_argument('--beta2', type=int, default=1, help='cutmix beta (default : 1)')
    parser.add_argument('--cutmix', type=bool, default=False, help='adjust cut mix option(dafault=False')
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
  
    args, _ = parser.parse_known_args()
    #parse_args()와 매우 유사하지만, 여분의 인자가 있을 때 error를 발생시키지 않고, 채워진 이름 공간에 여분의 인자의 문자열 리스트를 포함하는 두 항목을 튜플로 반환
    return args

if __name__ == '__main__':
    # Data and model checkpoints directories
    tuner_params = nni.get_next_parameter()
    #지정한 search_space에서 parameter를 가져옴
    logger_p.debug(tuner_params)

    params = vars(merge_parameter(get_params(),tuner_params))
    print(params)
    train(params)
