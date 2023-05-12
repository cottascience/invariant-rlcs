#!/bin/bash

#SBATCH -o deepsets_worker_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --array=11-201:20
echo $SLURM_ARRAY_TASK_ID
python main.py --dataset sort --model deepsets --lr 0.001 --input_size $SLURM_ARRAY_TASK_ID --hidden_size 25 --k 1 --m 1 --train_size 1000  > deepsets-$SLURM_ARRAY_TASK_ID.txt
