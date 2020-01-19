import torch.nn as nn
from config.data_utils import *

if(torch.cuda.is_available == True):
    class_weights=class_weights.cuda()
else:
    pass
loss = nn.CrossEntropyLoss(weight=class_weights)
def loss_fn(input,label):
    losses = loss(input,label)
    return losses