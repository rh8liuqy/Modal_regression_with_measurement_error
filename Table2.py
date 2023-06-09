from Beta_reg import *
from simulation_setting import *
simu_size = 1000

if __name__ == '__main__':
    simu_Table2(simu_size=simu_size,
                filename="01_Table2.csv",
                method="MCCL",
                n=200,
                SigmaW=np.sqrt(1.2),
                B=100)
    simu_Table2(simu_size=simu_size,
                filename="02_Table2.csv",
                method="MLE",
                n=200,
                SigmaW=np.sqrt(1.2),
                B=None)
