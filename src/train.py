import os
import ast 
from dataset import BengaliDatasetTrain
import torch
import torch.optim as optim
from tqdm import tqdm 
import torch.nn as nn 
from model_dispather import MODEL_DISPATCHER



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")
IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH"))

EPOCH = int(os.environ.get("EPOCH"))
TRAINING_BATCH_SIZE = int(os.environ.get("TRAINING_BATCH_SIZE"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))

MODEL_MEAN = ast.literal_eval(os.environ.get("MODEL_MEAN")) # it would be a list 
MODEL_STD = ast.literal_eval(os.environ.get("MODEL_STD"))

TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VAL_FOLDS = ast.literal_eval(os.environ.get("VAL_FOLDS"))

BASE_MODEL = os.environ.get("BASE_MODEL")

def loss_fn(outputs,targets):
    o1,o2,o3 = outputs
    t1,t2,t3 = targets
    l1 = nn.CrossEntropyLoss()(o1,t1)
    l2 = nn.CrossEntropyLoss()(o2,t2)
    l3 = nn.CrossEntropyLoss()(o3,t3)
    return torch.mean(l1+l2+l3)


def train(dataset,dataloader,model,optimizer):
    model.train()

    for bi,data in tqdm(enumerate(dataloader),total = int(len(dataset)/dataloader.batch_size)):
        images = data["image"]
        grapheme_root = data["grapheme_root"]
        vowel_diacritic = data["vowel_diacritic"]
        consonant_diacritic = data["consonant_diacritic"]

        images = images.to(DEVICE,dtype = torch.float)
        grapheme_root = grapheme_root.to(DEVICE,dtype = torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE,dtype = torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE,dtype =torch.long)

        optimizer.zero_grad()
        output = model(images)
        target = (grapheme_root,vowel_diacritic,consonant_diacritic)
        loss = loss_fn(output,target)
        loss.backward()
        optimizer.step()

def evaluate(dataset,dataloader,model):
    model.eval()
    final_loss = 0
    counter = 0
    for bi,data in tqdm(enumerate(dataloader),total = int(len(dataset)/dataloader.batch_size)):
        counter += 1
        images = data["image"]
        grapheme_root = data["grapheme_root"]
        vowel_diacritic = data["vowel_diacritic"]
        consonant_diacritic = data["consonant_diacritic"]

        images = images.to(DEVICE,dtype = torch.float)
        grapheme_root = grapheme_root.to(DEVICE,dtype = torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE,dtype = torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE,dtype =torch.long)

        output = model(images)
        target = (grapheme_root,vowel_diacritic,consonant_diacritic)
        loss = loss_fn(output,target)   
        final_loss += loss  
    return final_loss/counter 
print(torch.cuda.is_available())


def main():
    print("Device is ",DEVICE)
    model = MODEL_DISPATCHER[BASE_MODEL](pretrain=True)
    model.to(DEVICE)
    print("Model loaded !!! ") 
    train_dataset = BengaliDatasetTrain(
        folds=TRAINING_FOLDS,
        img_height = IMG_HEIGHT,
        img_width = IMG_WIDTH,
        mean = MODEL_MEAN,
        std = MODEL_STD
    )
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    valid_dataset = BengaliDatasetTrain(
        folds=VAL_FOLDS,
        img_height = IMG_HEIGHT,
        img_width = IMG_WIDTH,
        mean = MODEL_MEAN,
        std = MODEL_STD
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    optimizer = optim.Adam(model.parameters(),lr = 1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",patience=5,factor=0.3,verbose=True)

    if torch.cuda.device_count() > 1 :
        model = nn.DataParallel(model)

    for e in range(EPOCH):
        train(dataset=train_dataset,dataloader = train_loader,model = model,optimizer = optimizer)
        score = evaluate(dataset=valid_dataset,dataloader=valid_loader,model=model)
        scheduler.step(score)
        print("In the epoch {}, the loss in validation is {}".format(e,score))
        torch.save(model.state_dict(),"{}_fold{}.pth".format(BASE_MODEL,VAL_FOLDS[0]))



if __name__ == "__main__":
    main()     
