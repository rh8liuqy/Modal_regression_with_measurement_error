from Beta_reg import *
from simulation_setting import *
import sys

if __name__ == '__main__':
    filename = "Table4_" + "model" + sys.argv[1] + "n" + sys.argv[2]
    filename = filename + "seed" + sys.argv[3] + "_" + sys.argv[4] + ".csv"
    simu_Table4(seed_start=int(sys.argv[3]),
                seed_end=int(sys.argv[4]),
                filename=filename,
                model=sys.argv[1],
                n=int(sys.argv[2]),
                B=100)
