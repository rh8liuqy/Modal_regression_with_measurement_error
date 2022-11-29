#!/bin/sh
#SBATCH --job-name=s57
#SBATCH -N 1
#SBATCH -n 24    ##14 cores(of28) so you get 1/2 of machine RAM (64 GB of 128GB)
#SBATCH --gres=gpu:1   ## Run on 1 GPU
##SBATCH --output job%j.out
##SBATCH --error job%j.err
#SBATCH -p v100-16gb-hiprio


##Load your modules and run code here

module load cuda/10.0
module load python3/anaconda/2020.02
source activate /work/qingyang/test_env
python /work/qingyang/simu_beta/57simulation_N500_bootstrap_unknown_variance_probit.py
