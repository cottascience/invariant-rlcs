#!/bin/bash

#SBATCH -o rlc_worker_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --array=4-101:16
echo $SLURM_ARRAY_TASK_ID
python main.py --noise_size 4 --dataset ball --patience 30 --model rlc --input_size $SLURM_ARRAY_TASK_ID --hidden_size $SLURM_ARRAY_TASK_ID --train_size 1000 --lr 0.5 --batch_size 100 --k 10000 > ball/rlc-$SLURM_ARRAY_TASK_ID.txt
