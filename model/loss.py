import torch.nn as nn
loss = nn.CrossEntropyLoss()
def loss_fn(input,label):
    losses = loss(input,label)
    return losses