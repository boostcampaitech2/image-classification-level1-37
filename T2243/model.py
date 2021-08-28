from efficientnet_pytorch import EfficientNet

class MyModel(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b3', 
                                                in_channels=3, 
                                                num_classes=18) # weight가져오고 num_classes(두번째 파라미터로 학습시키는 class 수)
    
    def forward(self, x) :
        x = F.relu(self.model(x))
        return x