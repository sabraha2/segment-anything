#!/bin/bash

#$ -M sabraha2@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -l h=!qa-a10-*
#$ -e errors/
#$ -N finetune-mulit-sam-vit-base-100-epochs

# Required modules
module load conda
conda init bash
source activate sam

python finetune.py