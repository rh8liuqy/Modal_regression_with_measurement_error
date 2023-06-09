#!/bin/sh
#SBATCH --job-name=Figure2
#SBATCH --output job%j.out
#SBATCH --error job%j.err
#SBATCH -N 1
#SBATCH -n 48
#SBATCH -p defq-48core
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qingyang@email.sc.edu

##Load your modules first:
module load python3/anaconda/2020.02
source activate torch-env
##Add your code here:
hostname
date
cd /work/qingyang/beta_modal_regression
python Figure2.py 100 1 5000
python Figure2.py 200 1 5000
python Figure2.py 500 1 5000
python Figure2.py 1000 1 5000
