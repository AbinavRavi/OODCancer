from data_loader import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from utils.data_utils import all_classes
import tqdm
from model import model

seed = 137
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

metadata = './data/HAM10000_metadata.csv'
images = './data/'

train_data, val_data, _ = dataloader.prepare_data(metadata,all_classes[0],images,create_split=True,split=(0.7,0.2,0.1),batch=32)

#hyperparameters
epochs = 100
lr = 0.001
decay = 1e-4

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    lr_lambda=lambda step: cosine_annealing(step,
        5 * len(trainLoader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / 0.1))

model = model.ood_model(num_classes = 6)
optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=decay)


model.cuda()
trainLoss = []
valLoss = []
for i in tqdm.trange(epochs,desc='epochs',leave=False):
    losses = []
    model.train()
    for data,target in train_data:
        data,target = data.cuda(),target.cuda()
        x = model(data)

        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss(data,target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    trainLoss = np.array(losses).mean()
    model.eval()
    vallosses = []
    for vdata,vtarget in val_data:
        vdata,vtarget = vdata.cuda(), vtarget.cuda()
        x = model(data)

        vloss = nn.CrossEntropyLoss(vdata,vtarget)
        vallosses.append(vloss.item())
    valLoss = np.array(vallosses).mean()
    print('epoch:{} \t'.format(i+1),'trainloss:{}'.format(trainLoss),'\t','valloss:{}'.format(valLoss))
    if (epoch%5==0):
        torch.save(model,'./model/{}.pt'.format(i+1))
    
        








