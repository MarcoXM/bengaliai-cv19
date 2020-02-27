import albumentations as A
import pandas as pd 
import joblib
import numpy as np 
import torch
from PIL import Image


def strong_aug(img_height,img_width,mean,std,p=.5):
    return A.Compose([
        A.Resize(img_height,img_width,always_apply= True),
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),            
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Normalize(mean,std,always_apply= True)
    ])

class BengaliDatasetTrain:
    def __init__(self,folds, img_height, img_width, mean, std):
        df = pd.read_csv("../input/train_fols.csv")
        df = df[["image_id","grapheme_root", "vowel_diacritic", "consonant_diacritic","kfold"]]

        df = df[df.kfold.isin(folds)].reset_index(drop=True)
        self.image_ids = df.image_id.values[:1000]
        self.labels = df[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]].values
        self.image_data = pd.concat([pd.read_parquet(f"../input/train_image_data_{i}.parquet") for i in range(4)])
        self.train_aug = strong_aug(img_height, img_width, mean, std)

        if len(folds) ==1 :
            self.aug = A.Compose([
                A.Resize(img_height,img_width,always_apply= True),
                A.Normalize(mean,std,always_apply= True)
            ])
        else:
            self.aug = strong_aug(img_height, img_width, mean, std)


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,idx):
        image = self.image_data[self.image_data.image_id == self.image_ids[idx]].values[:,1:]
        image = image.reshape(137,236).astype(float)
        image = Image.fromarray(image).convert('RGB')
        image = self.aug(image=np.array(image))["image"]
        image = np.transpose(image,(2,0,1)).astype(np.float32)
        labels = self.labels[idx]
        return image,labels




