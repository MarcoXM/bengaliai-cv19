import pandas as pd 
import numpy as np 
import joblib
from tqdm import tqdm
import glob

if __name__ == "__main__":

    files = glob.glob("../input/train_*.parquet")
    for f in tqdm(files,total = len(files)):
        df = pd.read_parquet(f)
        image_ids = df.image_id.values
        df = df.drop("image_id",axis = 1)
        image_array = df.values

        for j,img_id in tqdm(enumerate(image_ids),total = len(image_ids)):
            print(img_id)
           # joblib.dump(image_array[j,:], f'../input/image_pickles/{img_id}.pkl')
