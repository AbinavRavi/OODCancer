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

test_data=dataloader.prepare_test_data(metadata, all_classes+['norm'], images)


#Load the model
model = torch.load('')

#helper functions
concat = lambda x:np.concatenate(x,axis=0)
to_np = lambda x: x.data.cpu().numpy()

def get_ood_scores(loader,in_classes):
    in_scores = []
    out_scores = []
    with torch.no_grad():
        for data,target in loader:
            data,target = data.cuda(),target.cuda()
            output = model(data)
            smax = F.softmax(output,1)
            for idx, trgt in enumerate(target):
                if trgt in in_classes:
                    in_scores.append(smax[idx])
                else:
                    out_scores.append(smax[idx])



    return in_scores, out_scores

in_score,out_score = get_ood_scores(test_data, all_classes[0:6])

auroc_list = []


def get_auroc(_pos, _neg):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)

    return auroc


# def get_and_print_results(ood_loader):
#     out_score = get_ood_scores(ood_loader)
#     auroc = get_auroc(out_score, in_score)
#     return auroc

#load ood data
auroc_score = get_auroc(out_score,in_score)
print(f'AUROC score: {auroc_score}')