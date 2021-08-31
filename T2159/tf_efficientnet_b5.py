import timm
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_model(num_classes):
    model = timm.create_model('tf_efficientnet_b5', pretrained=True, num_classes = num_classes)
    model.classifier = nn.Sequential(
    nn.Dropout(0.25),
    nn.Linear(2048, num_classes)
    )
    return model
