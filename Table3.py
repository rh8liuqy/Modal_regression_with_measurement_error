from Beta_reg import *
from simulation_setting import *
simu_size = 1000

if __name__ == '__main__':
    simu_Table3(simu_size=simu_size,
                filename="01_Table3.csv",
                known_variance=False,
                method="MCCL",
                n=200,
                B=100)
    simu_Table3(simu_size=simu_size,
                filename="02_Table3.csv",
                known_variance=True,
                method="MCCL",
                n=200,
                B=100)
    simu_Table3(simu_size=simu_size,
                filename="03_Table3.csv",
                known_variance=None,
                method="MLE",
                n=200,
                B=None)
