import torch.nn as nn

def loss_fn(input,label):
    loss = nn.MSELoss(input, label)
    return loss