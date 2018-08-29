#!/usr/bin/bash
#SBATCH -t 1-0
#SBATCH -n 1
#SBATCH --partition=gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1

# Run keras_classify.py on SLURM scheduling system

module add tensorflow python
#python3 keras_classify.py
python3 generator.py
