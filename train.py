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
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

metadata = './data/HAM10000_metadata.csv'
images = './data/'
batch=128
train_data, val_data, _ = dataloader.prepare_data(metadata,all_classes[1:],images,create_split=True,split=(0.64,0.16,0.2),batch=batch)

#hyperparameters
epochs = 100
lr = 1e-5
decay = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 

model = model.ood_model(num_classes = len(all_classes[1:]))
optimizer = optim.Adam(model.parameters(),lr=lr)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    lr_lambda=lambda step: cosine_annealing(step,
        5 * len(train_data),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / 0.1))

model.to(device)
trainLoss = []
valLoss = []
path='./logs/'
writer= SummaryWriter(f'{path}OOD_{lr}_{batch}')
for i in tqdm.trange(epochs,desc='epochs',leave=False):
    train_losses = []
    val_losses=[]

    model.train()
    for idx,(data,target) in enumerate(tqdm.tqdm(train_data,desc ='train_iter',leave=False)):
        # print(data.shape)
        # break
        #import pdb; pdb.set_trace()
        data,target = data.to(device),target.to(device)
        x = model(data)

        optimizer.zero_grad()
        loss = loss_fn(x,target)
        loss.backward()
        
        optimizer.step()
        #scheduler.step()
        
        train_losses.append(loss.item())
        writer.add_scalar('ItrLoss/train',loss.item(),i*len(train_data)+idx)
    trainLoss.append(np.array(train_losses).mean())
    
    model.eval()
    
    for idx,(vdata,vtarget) in enumerate(tqdm.tqdm(val_data,desc ='val_iter',leave=False)):
        vdata,vtarget = vdata.to(device), vtarget.to(device)
        y = model(vdata)
        optimizer.zero_grad()
        vloss = loss_fn(y,vtarget)
        val_losses.append(vloss.item())
        writer.add_scalar('ItrLoss/val',vloss.item(),i*len(val_data)+idx)
    valLoss.append(np.array(val_losses).mean())
    print('epoch:{} \t'.format(i+1),'trainloss:{0:.5f}'.format(trainLoss[i]),'\t','valloss:{0:.5f}'.format(valLoss[i]))

    writer.add_scalars('EpsLoss/',{'train':trainLoss[i],'val':valLoss[i]},i)
    
    if (i%2==0):
        torch.save(model,'./trained_models/{}_{}.pt'.format(lr,i+1))
    
    
        








