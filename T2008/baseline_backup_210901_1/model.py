import math
import timm
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from efficientnet_pytorch import EfficientNet
class MyModel(nn.Module):
    def __init__(self, num_classes, model_type, model_package):
        super().__init__()
        self.model_type = model_type
        self.model_package = model_package
        self.num_classes = num_classes
        if self.model_package.lower()=="timm":
            self.pretrain_model = timm.create_model('efficientnet_{self.model_type}', pretrained=True)
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
            
        elif self.model_package.lower() =="effnet":
            self.pretrain_model = EfficientNet.from_pretrained(f'efficientnet-{self.model_type}', num_classes=num_classes)
            # self.dropout = nn.Dropout(p=0.5)
        else:
            raise "NameError : Model package : timm, Effnet"

        


    def forward(self, x):
        if self.model_package.lower()=="effnet":
            x = self.pretrain_model(x)
            # x = F.silu(self.pretrain_model(x))
            # x = self.dropout(x)
            return x
        else:
            x = self.pretrain_model(x)
            x = self.add_layers(x)
            return x

# if __name__=="__main__":
#     tmp = MyModel(18)
#     import pdb;pdb.set_trace()
#     print(1)
#     print(1)
#     print(1)