import math
import timm
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from efficientnet_pytorch import EfficientNet

class EffnetModel(nn.Module):
    def __init__(self, num_classes, model_type):
        super().__init__()
        self.model_type = model_type
        self.num_classes = num_classes   
        self.pretrain_model = EfficientNet.from_pretrained(f'efficientnet-{self.model_type}', num_classes=num_classes)

    def forward(self, x):
            return self.pretrain_model(x)


class TimmModel(nn.Module):
    def __init__(self, num_classes, model_type):
        super().__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.pretrain_model = timm.create_model(f'efficientnet_{self.model_type}', pretrained=True)
        self.pretrain_model = nn.Sequential(*list(self.pretrain_model.children()))[:-2]
        last_num_feature = self.pretrain_model[5].num_features #for eff b4
        
        self.add_layers = nn.Sequential(OrderedDict([
            ('dropout1',nn.Dropout(p=0.5)),
            ('conv1', nn.Conv2d(last_num_feature, last_num_feature*2,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ('batch1', nn.BatchNorm2d(last_num_feature*2, eps = 1e-06, momentum = 0.1, affine=True, track_running_stats = True)),
            ('silu1', nn.SiLU(inplace=True)),
            ('dropout2',nn.Dropout(p=0.5)),
            ('conv2', nn.Conv2d(last_num_feature*2,last_num_feature,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
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

class sjmodel(nn.Module):
    def __init__(self,num_classes, model_type):
        super().__init__()
        self.model_type=model_type
        self.model_name = EfficientNet.from_pretrained(f'efficientnet-{self.model_type}',
                                                       in_channels=3,
                                                       num_classes=num_classes)
        if model_type=="b4":
            self.model_name._dropout = nn.Dropout(p=0.7, inplace=False)
        self.num_classes = num_classes
            
    def forward(self, x):
        x = F.relu(self.model_name(x))
        return x
    
    def param_init(self):
        self.model_name = EfficientNet.from_pretrained(f'efficientnet-{self.model_type}',
                                                       in_channels=3,
                                                       num_classes=self.num_classes)


class sumodel(nn.Module):
    def __init__(self, num_classes, model_type):
        super().__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.pretrain_model = timm.create_model(f'tf_efficientnet_{self.model_type}', pretrained=True, num_classes = num_classes)
        self.last_num_feature = self.pretrain_model.classifier.in_features
        self.pretrain_model.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.last_num_feature, num_classes))
        
    def forward(self,x):
        return self.pretrain_model(x)
 