import torch
import torch.nn as nn
import pandas as pd
import os
from data_dir import cfg_test
from dataloading import dataloaders
from model import MyModule_3, MyModule_2
from train import train
from eval import eval

torch.manual_seed(37)
if __name__ == "__main__":
    mask_model = MyModule_3()
    gender_model = MyModule_2()
    age_model = MyModule_3()

    models = {
        'mask' : mask_model,
        'gender' : gender_model,
        'age' : age_model
    }

    LEARNING_RATE = 0.0001
    NUM_EPOCH = 1000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device} is using")
    for category in ['mask','gender','age']:
        my_model = models[category]
        my_model.to(device)
        for parameter in my_model.parameters():
            parameter.requires_grad = False
        for parameter in my_model.my_model[1:].parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.Adam(my_model.parameters(), lr = LEARNING_RATE)
        loss_fn = nn.CrossEntropyLoss()
        dl = dataloaders()
        data_dict = dl.get_dict()
        train(category,my_model,NUM_EPOCH, optimizer,loss_fn,data_dict[category],cfg_test.data_dir,device)
    ans = eval([mask_model,gender_model,mask_model],device)
    submission = pd.read_csv(os.path.join(cfg_test.data_dir, 'info.csv'))
    submission['ans'] = ans
    submission.to_csv(os.path.join(cfg_test.data_dir, 'answer3'), index=False)
    print('test is done!')
    



