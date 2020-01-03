from data_loader.imageloader import data_loader_classifier
from data_loader.img_transform import *

from torch.utils.data import DataLoader
from torch.utils.data import random_split


seed=137
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def prepare_data(path_to_csv,load_classes,path_to_img,create_split=False,split=(70,10,20),batch=16):
    
    ds=data_loader_classifier(path_to_csv,load_classes, path_to_img)

        
    image_transform = Compose([
        ImToCaffe(),
        NpToTensor()
    ])
    target_transform = SegToTensor()
    
    
    ds = TransformData(ds, input_transforms=image_transform, target_transform=target_transform)

    if create_split:
        
        if sum(split) != 1:
            raise ValueError('Invalid Split. Sum of split to be 1')
       
        trainSize=int(split[0]*len(ds))
        valSize=int(split[1]*len(ds))
        testSize=int(len(ds)-trainSize-valSize)
        
        #split dataset here
        train_ds, val_ds, test_ds=random_split( ds,[trainSize,valSize,testSize])
        
        train_ds=DataLoader(train_ds, batch_size=batch, shuffle=True)
        val_ds=DataLoader(val_ds, batch_size=batch, shuffle=True)
        test_ds=DataLoader(train_ds, batch_size=batch, shuffle=True)
                
        return train_ds, val_ds, test_ds
    
    if not create_split:
        train_ds=DataLoader(ds, batch_size=batch, shuffle=True)
        return train_ds

