from torchvision.transforms import Compose
from torch.utils.data import Dataset
import torch 
import copy 
import numpy as np
from scipy import ndimage as ndi
from random import randint, uniform
from scipy import ndimage

def window_image(scan, window_center, window_width):
    
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    scan = np.clip(scan, img_min, img_max)
    
    return scan

class ImToCaffe(object):  
        
    def __call__(self, im):
        brain_img = window_image(im, 40, 80)
        subdural_img = window_image(im, 80, 200)
        soft_img = window_image(im, 40, 380)

        brain_img = (brain_img - 0) / 80
        subdural_img = (subdural_img - (-20)) / 200
        soft_img = (soft_img - (-150)) / 380
        bsb_img = np.array([brain_img, subdural_img, soft_img])
        
        return bsb_img

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
        seg = torch.from_numpy(seg.astype(np.float)).float()
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

    def __init__(self, dataset, input_transforms=None, target_transform=None):
        #super().__init__(dataset)
        self.ds = dataset
        self.input_transforms = input_transforms
        self.target_transform = target_transform
        

    def __getitem__(self, idx):
        # extract data from inner dataset
        inputs, target = self.ds[idx]
        #angle=randint(-10,10)
        translate=(randint(-15,15),randint(-15,15))
        #inputs=ndimage.rotate(inputs,angle,reshape=False,order=3)
        #target=ndimage.rotate(target,angle,reshape=False,order=1)
        inputs=ndimage.shift(inputs, translate,mode='nearest')
        target=ndimage.shift(target, translate,mode='nearest')
        tmax = target.max()
        target = ndi.binary_fill_holes(target/tmax,structure=np.ones((5,5)))
        target *= tmax
        
        inputs=self.input_transforms(inputs)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        
        # repackage data
        return inputs, target

    def __len__(self):
        return len(self.ds)
