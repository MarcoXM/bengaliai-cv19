import numpy as np 
import pandas as pd 
import gc
import albumentations as A # image augmentation
from albumentations.pytorch import ToTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import cv2


def prepare_image(data_type='train',
                  submission=True, indices=[0, 1, 2, 3]):
    assert data_type in ['train', 'test']
    if submission:
        image_df_list = [pd.read_parquet(f'{data_type}_image_data_{i}.parquet')
                         for i in indices]

    print('image_df_list', len(image_df_list))
    HEIGHT = 137
    WIDTH = 236
    images = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]
    del image_df_list
    gc.collect()
    images = np.concatenate(images, axis=0)
    return np.array(images)[:,:,:,np.newaxis]


class BengaliAIDataset(Dataset):
    def __init__(self, images, labels=None):
        super(BengaliAIDataset, self).__init__()
        self.images = images
        self.labels = labels
        self.tranforms2tensor = A.Compose([
            A.Resize(128,128,interpolation=cv2.INTER_AREA),
            A.Normalize(.5,.5),
                                          ToTensor()]
                                         )

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        image,label = self.images[idx],self.labels[idx]
        image = self.tranforms2tensor(image=image)['image']
        label = torch.as_tensor(label,dtype =torch.long)
        return image,label


class BengaliAITestDataset(Dataset):
    def __init__(self, images, labels=None):
        super(BengaliAITestDataset, self).__init__()
        self.images = images
        self.tranforms2tensor = A.Compose([
            A.Resize(128,128,interpolation=cv2.INTER_AREA),
            A.Normalize(.5,.5),
                                          ToTensor()]
                                         )

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        image= self.images[idx]
        image = self.tranforms2tensor(image=image)['image']
        return image


def get_loader(images,labels,loader_params,device='cpu'):
    dataset = BengaliAIDataset(images,labels)
    trainloader = DataLoader(dataset,**loader_params)
    batch_size = loader_params['batch_size']
    
    while True:
        for imgs,labs in trainloader:
            if batch_size != imgs.size(0):
                break
            else:
                imgs,labs = imgs.to(device),labs.to(device)
                imgs += (1.0/128.0) * torch.rand_like(imgs) # adding noise make model robust.
                
                yield imgs,labs

def get_testloader(images,loader_params,device='cpu'):
    dataset = BengaliAITestDataset(images)
    trainloader = DataLoader(dataset,**loader_params)
    return trainloader