import sys
sys.path.append('/work/qingyang/simu_beta')
import pybetareg as pyb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import os
from scipy.stats import norm

np.random.seed(300)

def b2(sample_size,m=3,repeated_measure=3,SigmaW=np.sqrt(1.2)):
    beta0 = 1
    beta1 = 1
    beta2 = 1
    output = np.zeros(sample_size*4)
    output = output.reshape(sample_size,4)
    for i in range(sample_size):
        Z1 = np.random.binomial(1,0.5,1)
        X1 = np.random.normal(((Z1 == 1)+0)-((Z1 == 0)),1,size=1)
        etaX = beta0 + beta1*(X1) + beta2*Z1
        thetaX = norm.cdf(etaX)
        Y = np.random.beta(1+m*thetaX,1+m*(1-thetaX),size=1)
        U1 = np.random.normal(0,SigmaW,1)
        U2 = np.random.normal(0,SigmaW,1)
        U3 = np.random.normal(0,SigmaW,1)
        W1 = X1 + U1
        W2 = X1 + U2
        W3 = X1 + U3
        SigmaWhat = np.std([W1,W2,W3],ddof=1)
        Wbar = np.mean([W1,W2,W3])
        output[i,] = np.column_stack((Y,Wbar,SigmaWhat,Z1))[0]
    output = pd.DataFrame(output, columns=['Y','Wbar','SigmaWhat','Z1'])
    return output

path = os.getcwd()

def hotelling_reject(seed,n=500):
    np.random.seed(seed)
    df1 = b2(n)
    y = df1['Y'].to_numpy()
    w = df1['Wbar'].to_numpy()
    z = df1['Z1'].to_numpy()
    z = np.column_stack([np.ones(z.shape[0]),z])
    sigmaw = df1['SigmaWhat'].to_numpy()
    model1 = pyb.reg(x=np.column_stack([z,w]), y=y,initial=[3,1,1,1])
    model1fit = model1.fit()
    initials = list(model1fit.params.values())
    initials = np.array(initials)[[0,3,1,2]]
    model2 = pyb.reg_measurement_error(y=y,w=w,z=z,
                                       sigmaw=sigmaw,
                                       initial=initials,
                                       CUDA = True,
                                       column_names = ['b1','b0','b2'])
    model2fit = model2.fit()
    if model2fit.message == "Optimization terminated successfully.":
        output = model2.hotelling_bootstrap(300)
        df_out = output.copy()
        df_out['seed'] = seed
        df_out = pd.DataFrame(df_out,index = [seed])
        filename = path + '/dir57/' + str(seed)+'.csv'
        df_out.to_csv(filename,index=False)
    else :
        output = {'p-value':np.nan}
    return output

def simu(seed):
    b = True
    while b:
        output = hotelling_reject(seed = seed)
        if np.isnan(output['p-value']):
            seed = seed + 5000
        else:
            b = False
    return output

if __name__ == '__main__':
    start_time = time.time()
    with mp.Pool(24) as p:
        output = p.map(simu, np.arange(0,300,1))
    print("--- %s seconds ---" % (time.time() - start_time))
    df_out = pd.DataFrame(output)
    df_out.to_csv("/work/qingyang/simu_beta/57simulation_N500_bootstrap_unknown_variance_probit.csv",index=False)

