#!/bin/bash

#SBATCH -o mlp_worker_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --array=8-32:2
echo $SLURM_ARRAY_TASK_ID
python main.py --dataset sin --patience 30 --model mlp --lr 0.01 --input_size $SLURM_ARRAY_TASK_ID --hidden_size $SLURM_ARRAY_TASK_ID  --k 1 --m 1 --train_size 1000 --batch_size 100 > ball/mlp-$SLURM_ARRAY_TASK_ID.txt
