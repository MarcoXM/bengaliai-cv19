import os
import ast 
from dataset import BengaliDatasetTrain
import torch
import torch.optim as optim
from tqdm import tqdm 
import torch.nn as nn 
from model_dispather import MODEL_DISPATCHER
import argparse
from distutils.util import strtobool
import os
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from numpy.random.mtrand import RandomState
from torch.utils.data.dataloader import DataLoader
from metrics import EpochMetric,macro_recall
from utils import create_evaluator,create_trainer,LogReport,ModelSnapshotHandler,output_transform


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")
IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH"))



WEIGHT_ONE=float(os.environ.get("WEIGHT_ONE"))
WEIGHT_TWO=float(os.environ.get("WEIGHT_TWO"))
WEIGHT_THR=float(os.environ.get("WEIGHT_THR"))

EPOCH = int(os.environ.get("EPOCH"))
TRAINING_BATCH_SIZE = int(os.environ.get("TRAINING_BATCH_SIZE"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))

MODEL_MEAN = ast.literal_eval(os.environ.get("MODEL_MEAN")) # it would be a list 
MODEL_STD = ast.literal_eval(os.environ.get("MODEL_STD"))

TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VAL_FOLDS = ast.literal_eval(os.environ.get("VAL_FOLDS"))

BASE_MODEL = os.environ.get("BASE_MODEL")
OUT_DIR = "../"



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
        shuffle=True,pin_memory=True,
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
        dataset = valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4,pin_memory=True
    )

    optimizer = optim.Adam(model.parameters(),lr = 1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-10)



    ## Define Trainer
    trainer = create_trainer(model, optimizer, DEVICE,WEIGHT_ONE,WEIGHT_TWO,WEIGHT_THR)

    # Recall for Training
    EpochMetric(
        compute_fn=macro_recall,
        output_transform=output_transform
    ).attach(trainer, 'recall')
    
    pbar = ProgressBar()
    pbar.attach(trainer, metric_names='all')

    evaluator = create_evaluator(model, DEVICE)

    #Recall for evaluating 
    EpochMetric(
        compute_fn=macro_recall,
        output_transform=output_transform
    ).attach(evaluator, 'recall')


    def run_evaluator(engine):
        evaluator.run(valid_loader)

    def schedule_lr(engine):
        # metrics = evaluator.state.metrics
        metrics = engine.state.metrics
        avg_mae = metrics['loss']

        # --- update lr ---
        lr = scheduler.optimizer.param_groups[0]['lr']
        scheduler.step(avg_mae)
        log_report.report('lr', lr)


        
    trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluator)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, schedule_lr)

    log_report = LogReport(evaluator, os.path.join(OUT_DIR,"log"))

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_report)
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        ModelSnapshotHandler(model, filepath=os.path.join(OUT_DIR,"weights","{}_fold{}.pth".format(BASE_MODEL,VAL_FOLDS[0]))))
    
    trainer.run(train_loader, max_epochs=EPOCH)


    train_history = log_report.get_dataframe()
    train_history.to_csv(os.path.join(OUT_DIR,"log","{}_fold{}_log.csv".format(BASE_MODEL,VAL_FOLDS[0])), index=False)

    print(train_history.head())
    print("Trainning Done !!!")
    


if __name__ == "__main__":
    main()     
