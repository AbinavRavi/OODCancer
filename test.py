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

seed = 137
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

metadata = './data/HAM10000_metadata.csv'
images = './data/'

_, _, test_data = dataloader.prepare_data(metadata,all_classes[1:],images,create_split=True,split=(0.7,0.1,0.2),batch=8)

#hyperparameters
epochs = 100
lr = 0.001
decay = 1e-4

#Load the model
model = torch.load('')

#helper functions
concat = lambda x:np.concatenate(x,axis=0)
to_np = lambda x: x.data.cpu().numpy()

def get_ood_scores(loader):
    _score = []
    scores_across_transforms = np.zeros(len(loader.dataset))
    with torch.no_grad():
        for t in range(num_perts):
            loader.dataset.pert_number = t
            start = 0
            for data,target in loader:
                data, target = data.cuda(), target.cuda()

                output = model(data)

                smax = F.softmax(output,1)

                # smax1 = F.softmax(output[:, :n_p1],1) # corresponds to softmax score for each transformation
                # smax2 = F.softmax(output[:, n_p1:n_p1 + n_p2],1)
                # smax3 = F.softmax(output[:, n_p1 + n_p2:],1)

                mask1 = torch.zeros_like(smax) 
                # mask2 = torch.zeros_like(smax2)
                # mask3 = torch.zeros_like(smax3)

                mask1.scatter_(1, target.view(-1, 1), 1.)
                # mask2.scatter_(1, t2.view(-1, 1), 1.)
                # mask3.scatter_(1, t3.view(-1, 1), 1.)

                score =  (smax1 * mask1).sum(1) #+ (smax2 * mask2).sum(1) + (smax3 * mask3).sum(1) 
                end = start+len(to_np(score))
                scores_across_transforms[start:end]+=to_np(score)
                start=end

    return -scores_across_transforms.copy()

in_score = get_ood_scores(test_data)

auroc_list = []


def get_auroc(_pos, _neg):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)

    return auroc


def get_and_print_results(ood_loader):
    out_score = get_ood_scores(ood_loader)
    auroc = get_auroc(out_score, in_score)
    return auroc

#load ood data
num_workers = 1
batch_size = 8
ood_scores = []

oodData = 