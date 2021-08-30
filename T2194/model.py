import torchvision
import torch
import torch.nn as nn
import math
from collections import OrderedDict
import timm


torch.manual_seed(37)

class myefficientnet_b0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pretrain_model = timm.create_model('tf_efficientnet_b0', pretrained=True,num_classes=18)
        self.pretrain_model.conv_head = nn.Sequential(nn.Conv2d(320, 640, kernel_size=(3, 3), stride=(1, 1), bias=False),nn.BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
                nn.SiLU(inplace=True),nn.Dropout(p=0.3),nn.Conv2d(640, 1280, kernel_size=(1, 1),
                stride=(1, 1), bias=False))    
    def forward(self,x):
        return self.pretrain_model(x)

class MSDefficientnetb4(nn.Module):
    def __init__(self,dropout_num=8,dropoutp=0.6):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4',
                                                  in_channels=3,
                                                  num_classes=18)  # weight가져오고 num_classes(두번째 파라미터로 학습시키는 class 수)
        self.model._dropout = nn.Identity()
        self.model._fc = nn.Identity()
        self.dropouts = nn.ModuleList([nn.Dropout(dropoutp) for _ in range(dropout_num)])
        self.fc = nn.Linear(in_features=1792, out_features=18, bias=True)

    def forward(self, x, y=None, loss_fn=None):
        feature = self.model(x)
        if len(self.dropouts) == 0:
            out = feature.view(feature.size(0), -1)
            out = self.fc(out)
            if loss_fn is not None:
                loss = loss_fn(out, y)
                return out, loss
            return out, None
        else:
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    out = dropout(feature)
                    out = out.view(out.size(0), -1)
                    out = self.fc(out)
                    if loss_fn is not None:
                        loss = loss_fn(out, y)
                else:
                    temp_out = dropout(feature)
                    temp_out = temp_out.view(temp_out.size(0), -1)
                    out = out + self.fc(temp_out)
                    if loss_fn is not None:
                        loss = loss + loss_fn(temp_out, y)
            if loss_fn is not None:
                return out / len(self.dropouts), loss / len(self.dropouts)
            return out, None
        return x

