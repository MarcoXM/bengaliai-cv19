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
from datetime import datetime
import os
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from numpy.random.mtrand import RandomState
from torch.utils.data.dataloader import DataLoader
from metrics import EpochMetric,macro_recall
from ignite.handlers import ModelCheckpoint, global_step_from_engine, EarlyStopping, TerminateOnNan
from utils import create_evaluator,create_trainer,LogReport,ModelSnapshotHandler,output_transform,get_lr_scheduler,get_optimizer


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

parameters={
        "alpha": 0.2,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "nesterov": True,
        "lr_max_value": 0.03,
        "lr_max_value_epoch": EPOCH // 15,
    }


def main():
    print("Device is ",DEVICE)
    model = MODEL_DISPATCHER[BASE_MODEL](pretrain=True)
    # model.load_state_dict(torch.load("../se_net/20200307-154303/weights/best_se_net_fold4_model_3_macro_recall=0.9435.pth"))
    model.to(DEVICE)
    print("Model loaded !!! ") 

    exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(os.path.join("../",BASE_MODEL)):
        os.mkdir(os.path.join("../",BASE_MODEL))
    OUT_DIR = os.path.join("../",BASE_MODEL,exp_name)
    print("This Exp would be save in ",OUT_DIR)

    os.mkdir(OUT_DIR)

    os.mkdir(os.path.join(OUT_DIR,"weights"))

    os.mkdir(os.path.join(OUT_DIR,"log"))


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


    optimizer = get_optimizer(
        model, 
        parameters.get("momentum"), 
        parameters.get("weight_decay"),
        parameters.get("nesterov")
    )
    lr_scheduler = get_lr_scheduler(
        optimizer, 
        parameters.get("lr_max_value"),
        parameters.get("lr_max_value_epoch"),        
        num_epochs=EPOCH,
        epoch_length=len(train_loader)
    )


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


    def get_curr_lr(engine):
        lr = lr_scheduler.schedulers[0].optimizer.param_groups[0]['lr']
        log_report.report('lr', lr)

    def score_fn(engine):
        score = engine.state.metrics['loss']
        return score
    es_handler = EarlyStopping(patience=30, score_function=score_fn, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, es_handler)
    def default_score_fn(engine):
        score = engine.state.metrics['recall']
        return score
    trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
    best_model_handler = ModelCheckpoint(dirname=os.path.join(OUT_DIR,"weights"),
                                        filename_prefix=f"best_{BASE_MODEL}_fold{VAL_FOLDS[0]}",
                                        n_saved=3,
                                        global_step_transform=global_step_from_engine(trainer),
                                        score_name="macro_recall",
                                        score_function=default_score_fn)
    evaluator.add_event_handler(Events.COMPLETED, best_model_handler, {"model": model, })
        
    trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluator)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, get_curr_lr)
    
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
