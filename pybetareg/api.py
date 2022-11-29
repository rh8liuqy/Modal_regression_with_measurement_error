"""
contains functions that user can use
"""
import numpy as np
import pybetareg.main

def reg(x,y,initial="NA",column_names="NA",link="logit"):
    x = np.array(x)
    y = np.array(y)
    out = pybetareg.main.betamodal(x,y,initial,column_names,link)
    return out

def reg_measurement_error(y,w,z,sigmaw,initial,
                          monte_carlo_size=100,
                          repeated_measure=3,
                          column_names="NA",
                          link="logit",
                          CUDA = False):
    y = np.array(y)
    w = np.array(w)
    z = np.array(z)
    sigmaw = np.array(sigmaw)
    initial = np.array(initial)
    out = pybetareg.main.betamodal_measurement_error(y,w,z,
                                                     sigmaw,
                                                     initial,
                                                     monte_carlo_size,
                                                     repeated_measure,
                                                     column_names,
                                                     link,
                                                     CUDA)
    return out