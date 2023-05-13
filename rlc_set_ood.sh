#!/bin/bash

#SBATCH -o rlc_set_ood_worker_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --array=5-101:2
echo $SLURM_ARRAY_TASK_ID
python main.py --dataset sort --model rlc_set --input_size $SLURM_ARRAY_TASK_ID --hidden_size 5 --train_size 1000 --OOD  > sort/rlc_set_ood-$SLURM_ARRAY_TASK_ID.txt