from PIL import Image
from torch.utils import data
import random
from torch.nn import functional as F
random.seed(10)
import os
import cv2
import numpy as np
import torch
try:
    from . import transform
except:
    import transform
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


#BGR
mean_rgb = np.array([[[0.551, 0.619, 0.532]]])*255
mean_t =np.array([[[0.341,  0.360, 0.753]]])*255
std_rgb = np.array([[[0.241, 0.236, 0.244]]])*255
std_t = np.array([[[0.208, 0.269, 0.241]]])*255


def getRandomSample(rgb,t):
    n = np.random.randint(10)
    zero = np.random.randint(2)
    if n==1:
        if zero:
            rgb = torch.from_numpy(np.zeros_like(rgb))
        else:
            rgb = torch.from_numpy(np.random.randn(*rgb.shape))
    elif n==2:
        if zero:
            t = torch.from_numpy(np.zeros_like(t))
        else:
            t = torch.from_numpy(np.random.randn(*t.shape))
    return rgb,t


class Data(Dataset):
    def __init__(self, root, mode='train'):
        self.samples = []
        lines = os.listdir(os.path.join(root, 'GT'))
        self.mode = mode
        for line in lines:
            rgbpath = os.path.join(root, 'RGB', line[:-4]+'.jpg')
            tpath = os.path.join(root, 'T', line[:-4]+'.jpg')
            maskpath = os.path.join(root, 'GT', line)
            self.samples.append([rgbpath,tpath,maskpath])


        if mode == 'test':
            self.transform = transform.Compose( 
                transform.Normalize(mean1=mean_rgb,mean2=mean_t,std1=std_rgb,std2=std_t),                             
                transform.Resize(288,288),                        
                transform.ToTensor())       
            
        else:
            raise ValueError

    def __getitem__(self, idx):
        rgbpath,tpath,maskpath = self.samples[idx]
        rgb = cv2.imread(rgbpath).astype(np.float32)
        t = cv2.imread(tpath).astype(np.float32)
        mask = cv2.imread(maskpath).astype(np.float32)
        H, W, C = mask.shape

        rgb,t,mask = self.transform(rgb,t,mask)   

        if self.mode == 'test':
            return rgb,t,mask, (H, W), maskpath.split('/')[-1]

                                       

    def __len__(self):
        return len(self.samples)

    
def get_loader(config, mode='train', pin=True):
    shuffle = False                 

    if mode == 'test':
        dataset = Data(config.test_root,mode='test')
        data_loader = DataLoader(dataset, batch_size=1,shuffle=False)
    return data_loader