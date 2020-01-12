from data_loader import dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from config.data_utils import all_classes
import tqdm.tqdm as tqdm
from model import model
from model import distillation_model
from model.loss import loss_fn
from model.distillation_loss import *
import pdb
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

metadata = './data/HAM10000_metadata.csv'
images = './data/'
batch=32
train_data, val_data, _ = dataloader.prepare_data(metadata,all_classes[1:],images,create_split=True,split=(0.7,0.1,0.2),batch=batch)

epochs = 100
lr = 0.0001
decay = 1e-4

# IN model name replace with trained model
model1 = torch.load('./trained_models/model_name.pt')
model2 = distillation_model.distill(input_size = 3, num_classes=len(all_classes[1:]))
optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=decay)

model1.eval()
model2.train()
train_loss = []
for i in range(epochs):
    train_losses = []
    for idx, (data,_) in enumerate(tqdm(train_data,desc='train_data',leave='False')):
        if (cuda.is_available() == True):
            data = data.cuda()
        else:
            data = data
    
        out1 = model1(data)
        out2 = model2(data)

        optimizer.zero_grad()
        loss = L2_loss(out1,out2)
        loss.backward()
        
        optimizer.step()
        train_losses.append(loss.item())
        writer.add_scalar('ItrLoss/train',loss.item(),i*len(train_data)+idx)

    train_loss.append(np.array(train_losses).mean())
    print('epoch:{} \t'.format(i+1),'trainloss:{0:.5f}'.format(trainLoss[i]))
    writer.add_scalars('EpsLoss/',{'train':trainLoss[i]},i)
    
    if (i%2==0):
        torch.save(model,'./trained_models/{}_{}.pt'.format(lr,i+1))





