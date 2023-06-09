from Beta_reg import *
from simulation_setting import *
import sys

if __name__ == '__main__':
    filename = "Figure2_" + "n" + sys.argv[1]
    filename = filename + "seed" + sys.argv[2] + "_" + sys.argv[3] + ".csv"
    simu_Figure2(seed_start=int(sys.argv[2]),
                 seed_end=int(sys.argv[3]),
                 filename=filename,
                 n=int(sys.argv[1]),
                 SigmaW=np.sqrt(1.2),
                 B=100)
