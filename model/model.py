from data_loader.dataloader import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torch.optim as optim
import copy

class ood_model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # pre-trained features
        backbone = models.resnet50(pretrained=True)
        
        backbone.fc = nn.Sequential(
               nn.Linear(2048, 512),
               nn.ReLU(inplace=True),
               nn.Dropout(p=0.5),
               nn.Linear(512, num_classes))

        backbone.conv1 = nn.Conv2d(3,64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.classifier=copy.deepcopy(backbone)

       
    def forward(self, x):
        x=self.classifier(x)
        return x