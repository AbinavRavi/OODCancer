import torch.nn as nn
from config.data_utils import class_weights

if(cuda.is_available == True):
    class_weights.cuda()
else:
    pass
loss = nn.CrossEntropyLoss(weight=class_weights)
def loss_fn(input,label):
    losses = loss(input,label)
    return losses