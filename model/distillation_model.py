import numpy as np
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F

class distill(nn.Module):
    def __init__(self, input_size,num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(input_size,16,kernel_size=2)
        self.conv2 = nn.Conv2d(16,32,kernel_size=2)
        self.conv3 = nn.Conv2d(32,64,kernel_size=2)
        self.conv4 = nn.Conv2d(64,128,kernel_size=2)
        self.conv5 = nn.Conv2d(128,256,kernel_size=2)
        self.fc = nn.Linear(256*219*219,num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        # pdb.set_trace()
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    