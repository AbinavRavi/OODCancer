from torchvision.transforms import Compose
from torch.utils.data import Dataset
import torch

import numpy as np
from scipy import ndimage as ndi
from random import randint, uniform
from scipy import ndimage

from config.data_utils import dsMean, dsStd 
from config.data_utils import all_classes

class ImToCaffe(object): 
    """
    Normalize the image; and convert images from [H,W,C]->[C,H,W]
    """
        
    def __call__(self, im):
        im=im.astype(float)
        im/=255.
        im=im-dsMean
        im=im/dsStd
        
        return im.transpose(2,0,1)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class NpToTensor(object):
    """
    Convert `np.array` to `torch.Tensor`, but not like `ToTensor()`
    from `torchvision` because we don't rescale the values.
    """

    def __call__(self, arr):
        return torch.from_numpy(np.ascontiguousarray(arr)).float()

    def __repr__(self):
        return self.__class__.__name__ + '()'

class SegToTensor(object):

    def __call__(self, seg):
        seg = seg
        return seg

    def __repr__(self):
        return self.__class__.__name__ + '()'
    

class TransformData(Dataset):
    """
    Transform a dataset by registering a transform for every input and the
    target. Skip transformation by setting the transform to None.

    Take
        dataset: the `Dataset` to transform (which must be a `SegData`).
        input_transforms: list of `Transform`s for each input
        target_transform: `Transform` for the target image
    """

    def __init__(self, dataset,load_classes, input_transforms=None, target_transform=None, ):
        #super().__init__(dataset)
        self.ds = dataset
        self.input_transforms = input_transforms
        self.target_transform = target_transform
        self.classes=load_classes
        
    def transform(self,inputs):
        angle=uniform(-20,20)
        translate=(uniform(-5,5),uniform(-5,5),0)
        flip=uniform(0,3)
        if flip<1:
            # flip horizontal
            inputs=np.flipud(inputs)
        else if flip>2:
            inputs=np.fliplr(inputs)
            # flip vertical
        inputs=ndimage.rotate(inputs,angle,reshape=False,order=3,mode='nearest')
        #import pdb; pdb.set_trace()
        inputs=ndimage.shift(inputs, translate,mode='nearest')

        return inputs


    def __getitem__(self, idx):
        # extract data from inner dataset
        inputs, target = self.ds[idx]
        target=self.classes.index(target)
        inputs = self.transform(inputs)
    
        inputs=self.input_transforms(inputs)
        return inputs, target

    def __len__(self):
        return len(self.ds)
