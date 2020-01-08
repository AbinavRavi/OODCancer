from data_loader.imageloader import data_loader_classifier
from data_loader.img_transform import *

from torch.utils.data import DataLoader
from torch.utils.data import random_split, SubsetRandomSampler

from torchvision import transforms
from sklearn.model_selection import train_test_split


seed=137
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def train_val_test_split(targets,split):
    
    train_idx, valid_idx= train_test_split(np.arange(len(targets)), 
                                                test_size=split[1]+split[2], shuffle=True, stratify=targets)

    valid_targets=[targets[idx] for idx in valid_idx]
    valid_idx, test_idx= train_test_split(np.arange(len(valid_targets)), 
                                                test_size=split[2]/(split[1]+split[2]), shuffle=True, stratify=valid_targets)
    #import pdb;pdb.set_trace()
    return train_idx, valid_idx, test_idx

def prepare_data(path_to_csv,load_classes,path_to_img,create_split=False,split=(70,10,20),batch=16):
    
    ds=data_loader_classifier(path_to_csv,load_classes, path_to_img)

        
    image_transform = Compose([
        ImToCaffe(),
        NpToTensor()
    ])

    target_transform = SegToTensor()
    
    
    ds = TransformData(ds,load_classes, input_transforms=image_transform, target_transform=target_transform)
    if create_split:
        
        if int(sum(split)) != 1:
            raise ValueError('Invalid Split. Sum of split to be 1')
        
        #split dataset here

        targets = ds.ds.cls
        train_idx, valid_idx, test_idx= train_val_test_split(targets,split)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        
        train_ds = torch.utils.data.DataLoader(dataset=ds, 
                                        batch_size=batch, 
                                        sampler=train_sampler)
        val_ds = torch.utils.data.DataLoader(dataset=ds, 
                                        batch_size=batch, 
                                        sampler=valid_sampler)
        test_ds = torch.utils.data.DataLoader(dataset=ds, 
                                        batch_size=batch, 
                                        sampler=test_sampler)
                
        return train_ds, val_ds, test_ds
    
    if not create_split:
        train_ds=DataLoader(ds, batch_size=batch, shuffle=True)
        return train_ds


