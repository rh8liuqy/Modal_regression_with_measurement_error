#!/bin/sh
#SBATCH --job-name=Table4
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
python Table4.py m2 200 1 300
python Table4.py m2 300 1 300
python Table4.py m2 400 1 300
python Table4.py m2 500 1 300
python Table4.py m3 200 1 300
python Table4.py m3 300 1 300
python Table4.py m3 400 1 300
python Table4.py m3 500 1 300
python Table4.py m4 200 1 300
python Table4.py m4 300 1 300
python Table4.py m4 400 1 300
python Table4.py m4 500 1 300
