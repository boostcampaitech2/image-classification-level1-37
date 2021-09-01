import math
import timm
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch

class MyModel1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pretrain_model = timm.create_model('tf_efficientnet_b4', pretrained=True,num_classes=num_classes)
        self.pretrain_model.conv_head = nn.Sequential(nn.Conv2d(448, 896, kernel_size=(3, 3), stride=(1, 1), bias=False),nn.BatchNorm2d(896, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                nn.SiLU(inplace=True),nn.Dropout(p=0.3),nn.Conv2d(896, 1792, kernel_size=(1, 1),
                stride=(1, 1), bias=False))  

    def forward(self,x):
        return self.pretrain_model(x)


class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.pretrain_model = timm.create_model('tf_efficientnet_b0', pretrained=True)
        self.pretrain_model = nn.Sequential(*list(self.pretrain_model.children()))[:-2]
        last_num_feature = self.pretrain_model[5].num_features
        self.add_layers = nn.Sequential(OrderedDict([
            ('dropout1',nn.Dropout(p=0.3)),
            ('conv1', nn.Conv2d(last_num_feature,2*last_num_feature,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ('batch1', nn.BatchNorm2d(2*last_num_feature, eps = 1e-06, momentum = 0.1, affine=True, track_running_stats = True)),
            ('silu1', nn.SiLU(inplace=True)),
            ('dropout2',nn.Dropout(p=0.3)),
            ('conv2', nn.Conv2d(2*last_num_feature,last_num_feature,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ('batch2', nn.BatchNorm2d(last_num_feature, eps = 1e-06, momentum = 0.1, affine=True, track_running_stats = True)),
            ('silu2',nn.SiLU(inplace=True)),
            ('avgfool',nn.AdaptiveAvgPool2d(output_size=(1,1))),
            ('flat', nn.Flatten()),
            ('fc', nn.Linear(last_num_feature,self.num_classes,bias=True)),
            
        ]))

        for _, layer in self.add_layers.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                stdv = 1./math.sqrt(layer.weight.size(1))
                layer.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.pretrain_model(x)
        x = self.add_layers(x)
        return x

class thick_model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.pretrain_model = timm.create_model('tf_efficientnet_b4',pretrained=True)
        self.add_model = list(self.pretrain_model.children())[3][4]
        self.pretrain_model_cut = nn.Sequential(*list(self.pretrain_model.children()))[:-2]
        self.model = nn.Sequential(OrderedDict([
            ('pretrained', self.pretrain_model_cut),
            ('dropout1', nn.Dropout(p=0.3)),
            ('conv1', nn.Conv2d(1792,448,kernel_size=(3,3),stride= (1,1),padding=(1,1),bias=False,)),
            ('batch1', nn.BatchNorm2d(448,eps=1e-06,momentum=0.1,affine=True,track_running_stats=True)),
            ('silu1', nn.SiLU(inplace=True)),
            ('dropout2', nn.Dropout(p=0.3)),
            ('conv2', nn.Conv2d(448,112,kernel_size=(3,3),stride= (1,1),padding=(1,1),bias=False,)),
            ('batch2', nn.BatchNorm2d(112,eps=1e-06,momentum=0.1,affine=True,track_running_stats=True)),
            ('silu2', nn.SiLU(inplace=True)),
            ('dropout3', nn.Dropout(p=0.3)),
            ('last', self.add_model),
            ('silu3',nn.SiLU(inplace=True)),
            ('dropout4', nn.Dropout(p=0.3)),
            ('conv3', nn.Conv2d(160,640,kernel_size=(3,3),stride= (1,1),padding=(1,1),bias=False,)),
            ('batch3', nn.BatchNorm2d(640,eps=1e-06,momentum=0.1,affine=True,track_running_stats=True)),
            ('silu3', nn.SiLU(inplace=True)),
            ('avgpool',nn.AdaptiveAvgPool2d(output_size=(1,1))),
            ('flat', nn.Flatten()),
            ('fc', nn.Linear(640,self.num_classes,bias=True))
        ]))
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for parameter in self.model[1:].parameters():
            parameter.requires_grad = True

        for _, layer in self.model[1:].named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                stdv = 1./math.sqrt(layer.weight.size(1))
                layer.bias.data.uniform_(-stdv, stdv)
        for _, layer in self.model.last.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                stdv = 1./math.sqrt(layer.weight.size(1))
                layer.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.model(x)
        return x

class front_model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.pretrain_model = timm.create_model('tf_efficientnet_b4',pretrained=True)
        self.head = nn.Sequential(*list(self.pretrain_model.children()))[:3]
        self.tail = nn.Sequential(*list(self.pretrain_model.children()))[3:-1]
        self.pretrain_model.classifier = nn.Linear(1792,num_classes,bias=True)
        self.full_model = nn.Sequential(OrderedDict([
            ('head', self.head),
            ('conv1', nn.Conv2d(48,192,kernel_size=(3,3),stride= (1,1),padding=(1,1),bias=False)),
            ('silu1', nn.SiLU(inplace=True)),
            ('dropout1', nn.Dropout(p=0.3)),
            ('conv2', nn.Conv2d(192,48,kernel_size=(3,3),stride= (1,1),padding=(1,1),bias=False)),
            ('silu2', nn.SiLU(inplace=True)),
            ('dropout2', nn.Dropout(p=0.3)),
            ('tail', self.tail),
            ('fc', nn.Linear(1792,self.num_classes,bias=True))
        ]))
        for parameter in self.full_model.parameters():
            parameter.requires_grad = True
        for _, layer in self.full_model[1:-1].named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                stdv = 1./math.sqrt(layer.weight.size(1))
                layer.bias.data.uniform_(-stdv, stdv)
        for _, layer in self.full_model.fc.named_modules():
            nn.init.xavier_uniform_(layer.weight)
            stdv = 1./math.sqrt(layer.weight.size(1))
            layer.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.pretrain_model(x)
        return x

if __name__ == "__main__":
    model = front_model(18)
    print(model.full_model)
