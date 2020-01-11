from torch.utils.data import Dataset

import pandas as pd
from skimage.io import imread
from skimage.transform import rescale

from config.data_utils import all_classes


class data_loader_classifier(Dataset):
    
    
    def __init__(self, path_to_csv, load_classes,path_to_img):
        
        df=pd.read_csv(path_to_csv)
        drop_classes=list(set(all_classes)-set(load_classes))
        
        for cls in drop_classes:
            df = df.drop(df[df.dx == cls].index)
            
        self.imgs=df['image_id'].tolist() 
        self.cls=df['dx'].tolist()
        self.path=path_to_img

    def get_image(self,path):
        img= imread(path)
        rescaled_img=rescale(img,0.25,order=3,multichannel=True)   

        return rescaled_img
    def __getitem__(self, idx):
        img_path=self.path+self.imgs[idx]+'.jpg'
        img=self.get_image(img_path)

        img_cls=self.cls[idx]
        return img, img_cls
        


    def __len__(self):
        return len(self.imgs)


    

