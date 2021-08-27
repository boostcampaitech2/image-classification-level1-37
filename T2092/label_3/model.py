import torchvision
import torch
import torch.nn as nn
import math
from collections import OrderedDict
torch.manual_seed(37)


class MyModule_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_model = nn.Sequential(OrderedDict([
            ('my_conv1', nn.Conv2d(2048, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ('batch', nn.BatchNorm2d(1024, eps = 1e-05, momentum = 0.1, affine=True, track_running_stats = True)),
            ('drop', nn.Dropout(p=0.5, inplace=True)),
            ('relu', nn.ReLU()),
            ('my_conv2', nn.Conv2d(1024, 2048, kernel_size = (3,3), stride=(1,1), padding=(1,1), bias = False)),
            ('batch2', nn.BatchNorm2d(2048, eps = 1e-05, momentum = 0.1, affine=True, track_running_stats = True)),
            ('relu2', nn.ReLU()),
            ('avgpool', nn.AdaptiveAvgPool2d(output_size = (1,1)))
        ]))
        self.fc_model = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(in_features = 2048, out_features = 2, bias = True))
        ]))
        self.pretrained =torchvision.models.resnet50(pretrained=True)
        self.pretrained = nn.Sequential(*list(self.pretrained.children())[:-2])
        self.my_model = nn.Sequential(
            self.pretrained,
            self.add_model,
            self.fc_model
        )
        for name, layer in self.my_model[1].named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
        for name, layer in self.my_model[2].named_modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                stdv = 1./math.sqrt(layer.weight.size(1))
                layer.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x):
        x = self.my_model[0](x)
        x = self.my_model[1](x)
        x= x.view(-1,2048)
        x = self.my_model[2](x)
        return x

class MyModule_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_model = nn.Sequential(OrderedDict([
            ('my_conv1', nn.Conv2d(2048, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ('batch', nn.BatchNorm2d(1024, eps = 1e-05, momentum = 0.1, affine=True, track_running_stats = True)),
            ('drop', nn.Dropout(p=0.5, inplace=True)),
            ('relu', nn.ReLU()),
            ('my_conv2', nn.Conv2d(1024, 2048, kernel_size = (3,3), stride=(1,1), padding=(1,1), bias = False)),
            ('batch2', nn.BatchNorm2d(2048, eps = 1e-05, momentum = 0.1, affine=True, track_running_stats = True)),
            ('relu2', nn.ReLU()),
            ('avgpool', nn.AdaptiveAvgPool2d(output_size = (1,1)))
        ]))
        self.fc_model = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(in_features = 2048, out_features = 3, bias = True))
        ]))
        self.pretrained =torchvision.models.resnet50(pretrained=True)
        self.pretrained = nn.Sequential(*list(self.pretrained.children())[:-2])
        self.my_model = nn.Sequential(
            self.pretrained,
            self.add_model,
            self.fc_model
        )
        for name, layer in self.my_model[1].named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
        for name, layer in self.my_model[2].named_modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                stdv = 1./math.sqrt(layer.weight.size(1))
                layer.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x):
        x = self.my_model[0](x)
        x = self.my_model[1](x)
        x= x.view(-1,2048)
        x = self.my_model[2](x)

        return x
