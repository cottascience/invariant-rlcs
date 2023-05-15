#!/bin/bash

#SBATCH -o rlc_worker_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --array=2-10:1
echo $SLURM_ARRAY_TASK_ID
python main.py --dataset sin --lr 0.05 --patience 30 --model rlc --input_size $SLURM_ARRAY_TASK_ID --hidden_size $SLURM_ARRAY_TASK_ID --k 10000 > ball/rlc-$SLURM_ARRAY_TASK_ID.txt

