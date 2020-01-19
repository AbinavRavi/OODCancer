from torch.utils.data import Dataset

import pandas as pd
from skimage.io import imread
from skimage.transform import resize

from config.data_utils import all_classes

import pickle

def load_pickle(path):
    with open(path,'rb') as handle:
        return pickle.load(handle)


class data_loader_classifier(Dataset):
    
    
    def __init__(self, path_to_csv, load_classes,path_to_img,path_to_pickle,test=False):
        
        df=pd.read_csv(path_to_csv)
        drop_classes=list(set(all_classes)-set(load_classes))
        
        for cls in drop_classes:
            df = df.drop(df[df.dx == cls].index)
            
        self.imgs=df['image_id'].tolist() 
        self.cls=df['dx'].tolist()
        self.path=path_to_img

        if test:
            norm_patients=load_pickle(path_to_pickle)
            print(len(norm_patients))
            self.imgs+=norm_patients
            self.cls+=['norm']*len(norm_patients)

        

    def get_image(self,path,img_cls):
        img= imread(path)
        if img_cls==7:
            img=img[:img.shape[0]//4,:img.shape[1]//4,:]
        rescaled_img=resize(img,(224,224,3),order=3)   

        return rescaled_img

    def __getitem__(self, idx):
        img_path=self.path+self.imgs[idx]+'.jpg'
        img_cls=self.cls[idx]
        img=self.get_image(img_path,img_cls)
        return img, img_cls
        


    def __len__(self):
        return len(self.imgs)


    

