#!/bin/bash
#SBATCH --job-name=dare_adaptive
#SBATCH --output=logs/dare_adaptive_%j.out
#SBATCH --error=logs/dare_adaptive_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=class
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hboddu12@asu.edu
REPO_DIR=/scratch/hboddu12/Perception_in_Robotics_DARE-1
PYTHON=/home/hboddu12/.conda/envs/env_dare/bin/python


cd $REPO_DIR
mkdir -p logs

module load cuda/12.9
module load cudnn/9.17.1-cuda12


echo "=============================="
echo "DARE Adaptive Scheduler Eval"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "GPU      : $CUDA_VISIBLE_DEVICES"
echo "Started  : $(date)"
echo "=============================="

/home/hboddu12/.conda/envs/env_dare/bin/python test_driver.py

echo "=============================="
echo "Finished : $(date)"
echo "=============================="
