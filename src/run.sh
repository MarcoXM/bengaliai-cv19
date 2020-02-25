#!/bin/bash
#SBATCH -J resnet34           
#SBATCH --cpus-per-task=8          
#SBATCH --gres=gpu:1          
#SBATCH -t 7-00:00:00

#SBATCH --mail-user=xwang423@fordham.edu
#SBATCH --mail-type=ALL
#SBATCH --output=stdout
#SBATCH --error=stderr
#SBATCH --exclude=node[001,002]
#SBATCH --nodes=1

module add cuda10.1/toolkit
module load shared
module load ml-pythondeps-py36-cuda10.1-gcc/3.0.0
module load pytorch-py36-cuda10.1-gcc/1.3.1

source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export IMG_HEIGHT=137
export IMG_WIDTH=236

export EPOCH=50
export TRAINING_BATCH_SIZE=32
export TEST_BATCH_SIZE=8

export MODEL_MEAN="(.485,.456,.406)"
export MODEL_STD="(.229,.224,.225)"

export BASE_MODEL="resnet34"
export TRAINING_FOLDS_CSV="../input/train_fols.csv"

export TRAINING_FOLDS="0,1,2,3"
export VAL_FOLDS="(4,)"
python train.py

export TRAINING_FOLDS="0,1,2,4"
export VAL_FOLDS="(3,)"
python train.py

#export TRAINING_FOLDS="0,1,3,4"
#export VAL_FOLDS="(2,)"
#python train.py

#export TRAINING_FOLDS="0,2,3,4"
#export VAL_FOLDS="(1,)"
#python train.py

#export TRAINING_FOLDS="1,2,3,4"
#export VAL_FOLDS="(0,)"
#python train.py
