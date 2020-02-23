import albumentations
import pandas as pd 
import joblib
import numpy as np 
import torch
from PIL import Image

class BengaliDatasetTrain:
    def __init__(self,folds, img_height, img_width, mean, std):
        df = pd.read_csv("../input/train_fols.csv")
        df = df[["image_id","grapheme_root", "vowel_diacritic", "consonant_diacritic","kfold"]]

        df = df[df.kfold.isin(folds)].reset_index(drop=True)
        self.image_ids = df.image_id.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritic.values
        self.consonant_diacritic = df.consonant_diacritic.values
        self.image_data = pd.concat([pd.read_parquet(f"../input/train_image_data_{i}.parquet") for i in range(4)])

        if len(folds) ==1 :
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height,img_width,always_apply= True),
                albumentations.Normalize(mean,std,always_apply= True)
            ])
        else:
            self.aug = albumentations.Compose([
                albumentations.Resize(img_height,img_width,always_apply= True),
                albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                                scale_limit=0.15,
                                                rotate_limit=5,
                                                p=0.9),
                albumentations.Normalize(mean,std,always_apply= True)])


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,idx):
        image = self.image_data[self.image_data.image_id == self.image_ids[idx]].values[:,1:]
        image = image.reshape(137,236).astype(float)
        image = Image.fromarray(image).convert('RGB')
        image = self.aug(image=np.array(image))["image"]
        image = np.transpose(image,(2,0,1)).astype(np.float32)

        return {
            "image": torch.tensor(image,dtype = torch.float32),
            "grapheme_root": torch.tensor(self.grapheme_root[idx],dtype=torch.long),
            "vowel_diacritic": torch.tensor(self.vowel_diacritic[idx],dtype=torch.long),
            "consonant_diacritic": torch.tensor(self.consonant_diacritic[idx],dtype=torch.long)
        }

