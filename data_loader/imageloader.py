from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import pickle
from scipy import ndimage
from skimage.io import imread

import nibabel as nib

import pydicom

def load_pickle(path):
    with open(path, 'rb') as handle:
           pickle_list=pickle.load(handle)
            
    return pickle_list 
            
from scipy.ndimage.interpolation import zoom
class data_loader_classifier(Dataset):
    
    
    def __init__(self, path_to_csv, path_to_dicom):
         pass
    

class data_loader_segmenter(Dataset):
    
    
    def __init__(self,path_pickle='./utils/', ct_dir = '/home/data/physionet-ct-isch/1.2.0/Raw_ct_scans/ct_scans/', mask_dir = '/home/data/physionet-ct-isch/1.2.0/Raw_ct_scans/masks/', ):
        
        self.ct_path=ct_dir
        self.mask_dir=mask_dir
        
        self.patient_list=load_pickle(path_pickle+'patient.pkl')
        self.disease_list=load_pickle(path_pickle+'anomaly.pkl')
        self.slice_list=load_pickle(path_pickle+'slice.pkl')
        
    def reshape_image(self,img):
        if img.shape!=(256,256):
            bfr_1= max(0,(256-img.shape[0])//2)
            afr_1= max(0,(256-img.shape[0])-bfr_1)
            bfr_2= max(0,(256-img.shape[1])//2)
            afr_2= max(0,(256-img.shape[1])-bfr_2)
            img=np.pad(img,((bfr_1,afr_1),(bfr_2,afr_2)),mode='edge')
            #print(img.shape)
            if img.shape!=(256,256):
            
                bfr_1= (img.shape[0]-256)//2
                afr_1= img.shape[0]-(img.shape[0]-256-bfr_1)
                bfr_2= (img.shape[1]-256)//2
                afr_2= img.shape[1]-(img.shape[1]-256-bfr_2)
                
                #print(bfr_1,afr_1,bfr_2,afr_2)
                img=img[bfr_1:afr_1,bfr_2:afr_2]
            
        #print(img.shape)    
        return img
        
    def get_img(self, pid,p_slc):
        if pid >99:
            scan_nii=nib.load(self.ct_path+str(pid)+'.nii')
        else:
            scan_nii=nib.load(self.ct_path+'0'+str(pid)+'.nii')
        #print(scan_nii.header['pixdim'][1:3])
        scan_img=scan_nii.get_fdata()[:,:,int(p_slc)-1]
        scan_img= ndimage.zoom(scan_img, scan_nii.header['pixdim'][1:3],order=3)
        scan_img=self.reshape_image(scan_img)
        #print(scan_img.shape)
        return scan_img, scan_nii.header['pixdim'][1:3]
    
    def get_label(self,pid,p_slc,disease_label,resize_factor):
        if disease_label==0:
            return np.zeros((256,256))
        else:
            if pid >99:
                label=imread(self.mask_dir+str(pid)+'/'+str(int(p_slc))+'_HGE_Seg.jpg')
            else:
                label=imread(self.mask_dir+'0'+str(pid)+'/'+str(int(p_slc))+'_HGE_Seg.jpg')
            label=label/255
            label=zoom(label.astype(np.int), resize_factor,order=1)
            
            label=self.reshape_image(label)*disease_label
            #print(np.unique(label))
            return label
            
        
    def __getitem__(self, idx):
        pid = self.patient_list[idx]
        
        #idx=idx+1
        pid = self.patient_list[idx]
        p_slc = self.slice_list[idx]
        
        scan,resize_factor=self.get_img(pid,p_slc)
        disease_label=self.disease_list[idx]
        label=self.get_label(pid,p_slc,disease_label, resize_factor)

   
        return  np.rot90(scan), label
    
    def __len__(self):
        return len(self.patient_list)
