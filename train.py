import pandas as pd 
import torch.optim as optim
from models import Dis
from data import BengaliAIDataset,get_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## id , its root, vowel, and consonant,
hyper = {'Dataloader': {'batch_size': 100, # first dimension
                      'shuffle':True}, # of course
         'Generator' : {'z_dim' : 100, # the length of z 

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




# data 
train = pd.read_csv('train.csv')
train_labels = train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values
indices = [0] if debug else [0, 1, 2, 3]
train_images = prepare_image(
    datadir = None, featherdir = None, data_type='train', submission=True, indices=indices)
trainloader = get_loader(train_images,train_labels,hyper['Dataloader'],device=device)

D = Dis(**hyper['Discriminator']).to(device, torch.float32)
print(D)
print('Model Initial !!!!')

optimizer_D = optim.Adam(params=D.parameters(), lr=hyper['lr_D'], betas=hyper['betas'])

decay_iter = hyper['num_iteration'] - hyper['decay_start'] # if your number of iteration is greater 
if decay_iter > 0:
    lr_lambda_D = lambda x: (max(0,1-x/(decay_iter*hyper['d_step'])))
    lr_sche_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lr_lambda_D)

# Training 

step = 1

while True:
    for i in range(hyper['d_step']): # 1
        for param in D.parameters():
            param.requires_grad_(True)
        #########
        #train D
        #########
        optimizer_D.zero_grad()
        
        #real image and related labels
        real_imgs, real_labels = next(iter(trainloader))
        batch_size = real_imgs.size(0)
        
        output = D(real_imgs) # real exam
        loss = calc_advloss_D(output,real_labels)
        loss.backward()
        optimizer_D.step()
        if (decay_iter > 0) and (step > hyper['decay_start']):
            lr_sche_D.step()
            
    if (decay_iter > 0) and (step > hyper['decay_start']):
        lr_sche_G.step()

    #print(loss_D.item(),loss_G.item())        
    if step < hyper['num_iteration']:
        step += 1
    else:
        print('total step: {}'.format(step))
        break
             
    if step%200 ==0:
        print('[%d/%d] Loss_D: %.4f'
                  % (step, hyper['num_iteration'],
                     loss.item()))
