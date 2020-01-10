from data_loader import dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from config.data_utils import all_classes
import tqdm
from model import model
from model.loss import loss_fn
import pdb

seed = 137
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

metadata = './data/HAM10000_metadata.csv'
images = './data/'

train_data, val_data, _ = dataloader.prepare_data(metadata,all_classes[1:],images,create_split=True,split=(0.7,0.1,0.2),batch=8)

#hyperparameters
epochs = 10
lr = 0.001
decay = 1e-4

model = model.ood_model(num_classes = len(all_classes[1:]))
optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=decay)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    lr_lambda=lambda step: cosine_annealing(step,
        5 * len(train_data),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / 0.1))




model.cuda()
trainLoss = []
valLoss = []
for i in tqdm.trange(epochs,desc='epochs',leave=False):
    losses = []
    model.train()
    for data,target in tqdm.tqdm(train_data,desc ='per_iteration',leave=False):
        # print(data.shape)
        # break
        data,target = data.cuda(),target.cuda()
        x = model(data)

        optimizer.zero_grad()
        # pdb.set_trace()
        loss = loss_fn(x,target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    trainLoss = np.array(losses).mean()
    model.eval()
    vallosses = []
    for vdata,vtarget in val_data:
        vdata,vtarget = vdata.cuda(), vtarget.cuda()
        y = model(vdata)
        vloss = loss_fn(y,vtarget)
        vallosses.append(vloss.item())
    valLoss = np.array(vallosses).mean()
    print('epoch:{} \t'.format(i+1),'trainloss:{}'.format(trainLoss),'\t','valloss:{}'.format(valLoss))
    if (epochs%5==0):
        torch.save(model,'./trained_models/{}.pt'.format(i+1))
    
        








