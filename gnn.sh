#!/bin/bash

#SBATCH -o gnn_worker_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --array=5-20:5
echo $SLURM_ARRAY_TASK_ID
python main.py --dataset connectivity --model gnn --k 1 --m 1 --lr 0.01 --input_size $SLURM_ARRAY_TASK_ID --hidden_size 2 --num_layers 3 --train_size 1000 --batch_size 100 > connectivity/gnn-$SLURM_ARRAY_TASK_ID.txt