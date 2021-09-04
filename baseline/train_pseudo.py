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
from tqdm import tqdm

from dataset import MaskBaseDataset
from loss import create_criterion
from sklearn.metrics import f1_score

import nni
import logging
from nni.utils import merge_parameter

""" pesudo labeling을 하는 트레이닝 파일입니다"""

logger_p = logging.getLogger('mask_AutoML')

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
        title = f"gt: {gt}, pred: {pred}"
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

def increment_path_sudo_path(path):
    if any(f.startswith("sudo") for f in os.listdir(path)):
        files = glob.glob('sub*.csv')
        files_number = [int(re.sub("[^\d+]","", file)) for file in files]
        max_number = max(files_number)
        return f"sudo_{max_number+1}.csv"
    else:
        return f"sudo_0.csv"

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
    train_loader = DataLoader(
        dataset,
        batch_size=args['batch_size'],
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
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
    if args['weights_type'] != None:
        train_data_labels = dataset.output_labels
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

    # -- pesudo labeling
    # create folder
    os.makedirs(args['output_dir'],exist_ok=True)

    ## file path
    img_root = os.path.join(args['data_dir_sudo'], 'images')
    info_path = os.path.join(args['data_dir_sudo'], 'info.csv')
    
    ###submission file path
    sub_filename = increment_path_sudo_path(args['output_dir'])
    sudo_path = os.path.join(args['output_dir'], sub_filename)

    #pesudo labeling dataset
    info = pd.read_csv(info_path)
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    TestDataset = getattr(import_module("dataset"), 'TestDataset')
    sudo_dataset = TestDataset(img_paths=img_paths,
                          height=args['resize_height'],
                          width=args['resize_width'])
    
    # sudo labeling activation function
    SoftMax=nn.Softmax(dim=1)

    # sudo labeling loader
    sudo_loader = torch.utils.data.DataLoader(
        sudo_dataset,
        batch_size=args['batch_size'],
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    best_f1 = 0
    best_acc = 0
    best_loss = np.inf
    for epoch in range(args['epochs']):
        # train loop
        model.train()
        f1_value_item = []
        loss_value_item = []
        matches_item = []
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if args['cutmix']==True:
                rand_bbox_module = getattr(import_module("utils"), 'rand_bbox')
                rand_bbox = rand_bbox_module(inputs=inputs,
                                            labels=labels,
                                            beta1=args['beta1'],
                                            beta2=args['beta2'],
                                            random_seed=args['seed'])
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
            f1_value_item.append(get_f1_score(labels,preds).item())
            loss_value_item.append(loss.item())
            matches_item.append((preds == labels).sum().item())
            if (idx + 1) % args['log_interval'] == 0:
                sudo_f1 = np.sum(f1_value_item) / (idx + 1)
                sudo_loss = np.sum(loss_value_item) / (idx + 1)
                train_acc = np.sum(matches_item) / args['batch_size'] / (idx + 1)
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch+1}/{args['epochs']}]({idx + 1}/{len(train_loader)}) || "
                    f"training f1 {sudo_f1:4.4} ||training loss {sudo_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/f1", sudo_f1, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/loss", sudo_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                
        
        f1 = np.sum(f1_value_item) / len(train_loader)
        loss = np.sum(loss_value_item) / len(train_loader)
        acc = np.sum(matches_item) / len(dataset)
        scheduler.step()
        

        if f1 > best_f1:
            print(f"New best model for train f1 score : {f1:4.2%}! saving the best model..")
            torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
            best_f1 = f1
            best_acc = acc
            best_loss = loss
        torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            
        print(
                f"[train] f1 : {f1:4.2%}, acc : {acc:4.2%}, loss: {loss:4.2} || "
                f"best f1 : {best_f1:4.2%}, best acc : {best_acc:4.2%}, best loss: {best_loss:4.2}"
        )
        logger.add_scalar("train/f1", f1, epoch)
        logger.add_scalar("train/loss", loss, epoch)
        logger.add_scalar("train/accuracy", acc, epoch)
        
        nni.report_intermediate_result(best_f1)
        logger_p.debug('test accuracy(train) is %g', f1)
        logger_p.debug('test accuracy(best) is %g', best_f1)
    
    logger_p.debug('Final train result(f1 score) is %g', best_f1)
    logger_p.debug('Send final train result done.')
    
    print("Calculating pesudo labeling..")
    for i in range(args['epochs_SL']):
        model.eval()
        preds = []
        with torch.no_grad():
            for idx, images in enumerate(tqdm(sudo_loader)):
                images = images.to(device)
                pred = model(images)
                prob=SoftMax(pred)
                prob_bool=torch.argmax(prob,dim=-1)>=args['threshold']
                prob_argmax=torch.argmax(prob,dim=-1)
                for j in range(len(prob_bool)):
                    if prob_bool[j]:
                        preds.append(prob_argmax[j].cpu().numpy())
                    else:
                        preds.append(100)

        info['ans'] = preds
        info.to_csv(sudo_path, index=False)

        
        info_train = pd.read_csv(sudo_path)
        info_train = info_train[info_train.ans!=100]
        img_paths_train = [os.path.join(img_root, img_id) for img_id in info_train.ImageID]
        img_labels_train = [int(label) for label in info_train.ans]
        sudo_dataset_train = TestDataset(img_paths=img_paths_train,
                                        img_labels=img_labels_train,
                                        height=args['resize_height'],
                                        width=args['resize_width'])

        sudo_loader_train = torch.utils.data.DataLoader(sudo_dataset_train,
                                                        num_workers=multiprocessing.cpu_count()//2,
                                                        batch_size=args['batch_size'],
                                                        shuffle=False,
                                                        pin_memory=use_cuda,
                                                        drop_last=False)
        model.train()
        for epoch in range(i+2): # sudo-labeling 누적 시행 횟수+2씩 sudo labeling data로 추가 학습
            f1_value_item = []
            loss_value_item = []
            matches_item = []
            for idx, train_batch in enumerate(sudo_loader_train):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)
            
                if args['cutmix']==True:
                    rand_bbox_module = getattr(import_module("utils"), 'rand_bbox')
                    rand_bbox = rand_bbox_module(inputs=inputs,
                                                labels=labels,
                                                beta1=args['beta1'],
                                                beta2=args['beta2'],
                                                random_seed=args['seed'])
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
                f1_value_item.append(get_f1_score(labels,preds).item())
                loss_value_item.append(loss.item())
                matches_item.append((preds == labels).sum().item())
                if (idx + 1) % args['log_interval'] == 0:
                    sudo_f1 = np.sum(f1_value_item) / (idx + 1)
                    sudo_loss = np.sum(loss_value_item) / (idx + 1)
                    sudo_acc = np.sum(matches_item) / args['batch_size'] / (idx + 1)
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch+1}/{args['epochs']}]({idx + 1}/{len(sudo_loader_train)}) || "
                        f"training f1 {sudo_f1:4.4} ||training loss {sudo_loss:4.4} || training accuracy {sudo_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/f1", sudo_f1, epoch * len(sudo_loader_train) + idx)
                    logger.add_scalar("Train/loss", sudo_loss, epoch * len(sudo_loader_train) + idx)
                    logger.add_scalar("Train/accuracy", sudo_acc, epoch * len(sudo_loader_train) + idx)
                
        
            f1 = np.sum(f1_value_item) / len(sudo_loader_train)
            loss = np.sum(loss_value_item) / len(sudo_loader_train)
            acc = np.sum(matches_item) / len(sudo_dataset_train)
            scheduler.step()

            if f1 > best_f1:
                print(f"New best model for val f1 score : {f1:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_f1 = f1
                best_acc = acc
                best_loss = loss
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")   

            print(
                f"[train] f1 : {f1:4.2%}, acc : {acc:4.2%}, loss: {loss:4.2} || "
                f"best f1 : {best_f1:4.2%}, best acc : {best_acc:4.2%}, best loss: {best_loss:4.2}"
            )       
            logger.add_scalar("train/f1", f1, epoch)
            logger.add_scalar("train/loss", loss, epoch)
            logger.add_scalar("train/accuracy", acc, epoch)
        
            nni.report_intermediate_result(best_f1)
            logger_p.debug('test accuracy(train) is %g', f1)
            logger_p.debug('test accuracy(best) is %g', best_f1)
    
    nni.report_final_result(best_f1)
    logger_p.debug('Final result(f1 score) is %g', best_f1)
    logger_p.debug('Send final sudo beling done.')
            


def get_params():
    #argparse를 사용하기 위해 argumentparser 객체 생성
    parser = argparse.ArgumentParser(description='Mask')

    parser.add_argument('--seed', type=int, default=37, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskBaseDataset)') #
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation_Max', help='data augmentation type (default: BaseAugmentation)') #CustomAugmentation
    parser.add_argument('--resize_height', type=int, default=512, help='input resize_height size for resize augmentation (default: 300)')# b0 224 ,b1 240 ,b2 260 ,b3 300 
    parser.add_argument('--resize_width', type=int, default=384, help='input resize_width size for resize augmentation (default: 300)') # b4 380 ,b5 456 ,b6 528 ,b7 600 
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='EffnetModel', help='model class name in model.py(EffnetModel, TimmModel)')
    parser.add_argument('--model_type', type=str, default='b4', help='model type (range: b0~b7)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--reduction', type=str, default='sum', help='criterion reduction type (default: mean)')
    parser.add_argument('--lr_decay_step', type=int, default=100, help='learning rate scheduler decay step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=50, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--weights_type', type=str, default='None', help=' weights type (default: None)')
    parser.add_argument('--beta1', type=int, default=1, help='cutmix beta (default : 1)')
    parser.add_argument('--beta2', type=int, default=1, help='cutmix beta (default : 1)')
    parser.add_argument('--cutmix', type=bool, default=False, help='adjust cut mix option(dafault=True')
    parser.add_argument('--threshold', type=float, default=0.7, help='sudo labeling threshold (default: 0.7)')
    parser.add_argument('--epochs_SL', type=int, default=3, help='epochs for sudo_labelling (default: 3)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--data_dir_sudo', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
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
