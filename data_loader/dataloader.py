from utils.imageloader import data_loader_classifier
from utils.imageloader import data_loader_segmenter

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from utils.medTransform import *

seed=137
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def prepare_data(path_pickle='./utils/', ct_dir = '/home/data/physionet-ct-isch/1.2.0/Raw_ct_scans/ct_scans/', mask_dir = '/home/data/physionet-ct-isch/1.2.0/Raw_ct_scans/masks/', batch=1,create_split=True, split=(0.7,.15,.15), segmenter=True):
    if segmenter:
        dataloader= data_loader_segmenter(path_pickle,ct_dir, mask_dir)
    
    
    if not segmenter:
        raise('Not Done!!')
        
    image_transform = Compose([
        ImToCaffe(),
        NpToTensor()
    ])
    target_transform = SegToTensor()
    
    # Medical data to 3 dim data like other image files
    ds = TransformData(dataloader, input_transforms=image_transform, target_transform=target_transform)

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

