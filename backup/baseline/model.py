import math
import timm
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# class BaseModel(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.25)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)

#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)

#         x = self.conv3(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout2(x)

#         x = self.avgpool(x)
#         x = x.view(-1, 128)
#         return self.fc(x)


# # Custom Model Template
# class MyModel(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         """
#         1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
#         2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
#         3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
#         """

#     def forward(self, x):
#         """
#         1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
#         2. 결과로 나온 output 을 return 해주세요
#         """
#         return x

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.pretrain_model = timm.create_model('tf_efficientnet_b0', pretrained=True)
        self.pretrain_model = nn.Sequential(*list(self.pretrain_model.children()))[:-3]
        last_num_feature = self.pretrain_model[5].num_features
        self.add_layers = nn.Sequential(OrderedDict([
            ('my_dropout1',nn.Dropout(p=0.3)),
            ('my_conv1', nn.Conv2d(last_num_feature,2*last_num_feature,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ('batch1', nn.BatchNorm2d(2*last_num_feature, eps = 1e-06, momentum = 0.1, affine=True, track_running_stats = True)),
            ('my_silu2', nn.SiLU(inplace=True)),
            ('my_dropout2',nn.Dropout(p=0.3)),
            ('my_conv2', nn.Conv2d(2*last_num_feature,last_num_feature,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)),
            ('batch2', nn.BatchNorm2d(last_num_feature, eps = 1e-06, momentum = 0.1, affine=True, track_running_stats = True)),
            ('my_silu2',nn.SiLU(inplace=True)),
            ('avgfool',nn.AdaptiveAvgPool2d(output_size=(1,1))),
            ('flat', nn.Flatten()),
            ('my_fc', nn.Linear(last_num_feature,self.num_classes,bias=True)),
            

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

# if __name__=="__main__":
    # tmp = MyModel(18)
    # print(tmp.pretrain_model)
    # print(tmp.add_layers)

