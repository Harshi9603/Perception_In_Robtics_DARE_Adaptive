#!/bin/bash

#SBATCH -N 1
#SBATCH -c 24
#SBATCH -t 5:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH -p general
#SBATCH -q class
#SBATCH -o /scratch/hboddu12/dare_joblogs/slurm.%j.out
#SBATCH -e /scratch/hboddu12/dare_joblogs/slurm.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="hboddu12@asu.edu"


module purge
# (Optional) Load CUDA if required by cluster
module spider cuda
# module load cuda/12.x   # uncomment if needed

# Initialize conda properly
source ~/.bashrc
conda activate env_dare
module load mamba/latest


cd /scratch/hboddu12/Perception_in_Robotics_DARE-1

export WANDB_API_KEY='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'

python k_sampling_reranking_driver.py


