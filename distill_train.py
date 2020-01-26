from data_loader import dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from config.data_utils import all_classes
from tqdm import tqdm
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
batch=16
train_data, val_data, _ = dataloader.prepare_data(metadata,all_classes[1:],images,create_split=True,split=(0.7,0.1,0.2),batch=batch)

epochs = 100
lr = 0.001
decay = 1e-4
path='./distill_logs/'
writer = SummaryWriter(f'{path}OOD_distill_{lr}_{batch}')
# IN model name replace with trained model
model1 = torch.load('./trained_models/1e-05_61.pt')
model2 = distillation_model.distill(input_size = 3, num_classes=len(all_classes[1:]))
lfn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model2.parameters(),lr=lr,weight_decay=decay)
  
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5)

model1.eval()
model2.train()
model2.cuda()
train_loss = []
c_loss = []
for i in tqdm(range(epochs)):
    train_losses = []
    cl_losses = []
    for idx, (data,target) in enumerate(tqdm(train_data,desc='train_data',leave='False')):
        # if (torch.cuda.is_available() == True):
        #     data = data.cuda()
        # else:
        #     data = data
        data = data.cuda()
        target = target.cuda()
    
        out1 = model1(data)
        out2 = model2(data)

        optimizer.zero_grad()
        loss = L2_loss(out1,out2)
        cl_loss = lfn(out2,target)
        loss.backward()
        
        optimizer.step()
        # scheduler.step(train_loss)
        train_losses.append(loss.item())
        cl_losses.append(cl_loss.item())
        writer.add_scalar('ItrLoss/train',loss.item(),i*len(train_data)+idx)
        writer.add_scalar('ItrLoss_class/train',cl_loss.item(),i*len(train_data)+idx)

    train_loss.append(np.array(train_losses).mean())
    c_loss.append(np.array(cl_losses).mean())
    scheduler.step(train_loss[i])
    print('epoch:{} \t'.format(i+1),'trainloss:{0:.5f}'.format(train_loss[i]),'classifier_loss:{0:.5f}'.format(c_loss[i]))
    for param_group in optimizer.param_groups:
        writer.add_scalar('Learning rate',param_group['lr'],i)
    writer.add_scalars('EpsLoss/',{'train':train_loss[i],'classifier':c_loss[i]},i)
    
    if (i%2==0):
        torch.save(model2,'./trained_models/L1_distilled_{}_{}.pt'.format(lr,i+1))





