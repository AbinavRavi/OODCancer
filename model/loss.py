import torch.nn as nn

def loss_fn(input,label):
    loss = nn.CrossEntropyLoss(input, label)
    return loss