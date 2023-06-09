from Beta_reg import *
from simulation_setting import *
simu_size = 1000

if __name__ == '__main__':
    simu_Figure1(simu_size=simu_size,
                 filename="01_Figure1.csv",
                 indepedent=False,
                 method="MCCL",
                 n=2000,
                 SigmaW=np.sqrt(3),
                 B=100)
    simu_Figure1(simu_size=simu_size,
                 filename="02_Figure1.csv",
                 indepedent=True,
                 method="MCCL",
                 n=2000,
                 SigmaW=np.sqrt(3),
                 B=100)
    simu_Figure1(simu_size=simu_size,
                 filename="03_Figure1.csv",
                 indepedent=False,
                 method="MLE",
                 n=2000,
                 SigmaW=np.sqrt(3),
                 B=None)
    simu_Figure1(simu_size=simu_size,
                 filename="04_Figure1.csv",
                 indepedent=True,
                 method="MLE",
                 n=2000,
                 SigmaW=np.sqrt(3),
                 B=None)
    