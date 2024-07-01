#!/bin/bash

#$ -M sabraha2@nd.edu
#$ -m abe
#$ -q gpu
#$ -l gpu_card=1
#$ -l h=qa-rtx6k*
#$ -e errors/
#$ -N finetune-mulit-sam-vit-base-50-epochs

# Required modules
module load conda
conda init bash
conda activate segment-anything-env

python finetune.py