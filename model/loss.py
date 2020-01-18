import torch.nn as nn
from config.data_utils import class_weights
import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if(torch.cuda.is_available == True):
    # class_weights.to(device)
# else:
    # pass
class_weights = class_weights.cuda()

loss = nn.CrossEntropyLoss(weight=class_weights)
def loss_fn(input,label):
    losses = loss(input,label)
    return losses