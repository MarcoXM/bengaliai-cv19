import torch
from models import Dis
import os
import pandas as pd
from scipy import stats
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import joblib
from data import prepare_image, get_loader,get_testloader


hyper = {'testloader': {'batch_size': 100, # first dimension
                      'shuffle':False}, # of course

         'Discriminator': {'ch': 64, #same
                          'n_grapheme': 168,
                           'n_vowel': 11,
                           'n_consonant':7, # same
                          'use_attn': True},
         'num_iteration': 40000, # train 26000 iter
         'decay_start' : 35000, # getting the result from graph
         'd_step' : 1, 
         'lr_D': 4e-4,
         'betas': (0.0, # adam hyperparameter
                  0.999),
         'margin': 1.0, # loss trade off
         'gamma':0.1, 
         'ema': 0.999,
         'seed': 224} # my favorite seed


device = 'cuda' if torch.cuda.is_available() else 'cpu'
debug=False
indices = [0] if debug else [0, 1, 2, 3]
test_images = prepare_image(data_type='test', submission=True, indices=indices)
testloader = get_testloader(test_images,hyper['testloader'],device=device)


D = Dis(**hyper['Discriminator']).to(device, torch.float32)
D.load_state_dict(torch.load('model_40000.pt'))


D.eval()
preds_dict = {
    'grapheme_root': [],
    'vowel_diacritic': [],
    'consonant_diacritic': []
}
components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
target=[] # model predictions placeholder
row_id=[] # row_id place holder
for i,data in enumerate(testloader): 
    data = data.to(device)
    with torch.no_grad():
        out1,out2,out3 = D(data)
        _, pred1 = torch.max(out1, 1)
        _, pred2 = torch.max(out2, 1)
        _, pred3 = torch.max(out3, 1)
        preds = (pred1.cpu().numpy(),pred2.cpu().numpy(),pred3.cpu().numpy())
        for i, p in enumerate(preds_dict):
            preds_dict[p] = preds[i]
        for i in range(len(test_images)):  
            for j,comp in enumerate(components):
                id_sample='Test_'+str(i)+'_'+comp
                row_id.append(id_sample)
                target.append(preds_dict[comp][i])

df_sample = pd.DataFrame(
    {
        'row_id': row_id,
        'target':target
    },
    columns = ['row_id','target'] 
)
df_sample.to_csv('submission.csv',index=False)
