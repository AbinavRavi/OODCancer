import torch
import torch.nn as nn

def L2_loss(l1,l2):
    loss = nn.MSELoss()
    losses = loss(l1,l2)
    return losses

def L1_loss(l1,l2):
    loss = nn.L1Loss()
    losses = loss(l1,l2)
    return losses

def cosine(l1,l2):
    loss = nn.CosineSimilarity()
    losses = loss(l1,l2)
    return losses

