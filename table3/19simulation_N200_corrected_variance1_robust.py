import sys
sys.path.append('C:\\Users\\Kevin_Liu\\OneDrive - University of South Carolina\\Research\\Beta Measurement Error\\paper\\simulation')
import pybetareg as pyb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import time

np.random.seed(300)

def b2(sample_size,m=3,repeated_measure=3,SigmaW=np.sqrt(3)):
    beta0 = 0.25
    beta1 = 0.25
    beta2 = 0.25
    output = np.zeros(sample_size*5)
    output = output.reshape(sample_size,5)
    for i in range(sample_size):
        Z1 = np.random.binomial(1,0.5,1)
        X1 = np.random.normal(((Z1 == 1)+0)-((Z1 == 0)),1,size=1)
        etaX = beta0 + beta1*(X1) + beta2*Z1
        thetaX = 1/(1+np.exp(-etaX))
        Y = np.random.beta(1+m*thetaX,1+m*(1-thetaX),size=1)
        Wbar = X1+np.random.laplace(0,np.sqrt(1/2),size=1)
        SigmaW = SigmaW
        output[i,] = np.column_stack((Y,Wbar,SigmaW,Z1,X1))[0]
    output = pd.DataFrame(output, columns=['Y','Wbar','SigmaW','Z1','X1'])
    return output

def simu(seed,n=200):
    np.random.seed(seed)
    df1 = b2(n)
    y = df1['Y'].to_numpy()
    w = df1['Wbar'].to_numpy()
    z = df1['Z1'].to_numpy()
    z = np.column_stack([np.ones(z.shape[0]),z])
    sigmaw = df1['SigmaW'].to_numpy()
    model2 = pyb.reg_measurement_error(y=y,w=w,z=z,
                                       sigmaw=sigmaw,
                                       initial=[3,0.25,0.25,0.25],
                                       CUDA = True,
                                       column_names = ['b1','b0','b2'])
    model2fit = model2.fit()
    dict1 = model2fit.params
    cov = np.sqrt(np.diag(model2fit.params_cov))
    dict2 = {'m_cov':cov[0],
             'b1_cov':cov[1],
             'b0_cov':cov[2],
             'b2_cov':cov[3],
             'message':model2fit.message}
    dict1.update(dict2)
    return dict1

if __name__ == '__main__':
    start_time = time.time()
    with mp.Pool(3) as p:
        output = p.map(simu, np.arange(0,1000,1))
    df_out = pd.DataFrame(output)
    index = df_out['message'] == 'Optimization terminated successfully.'
    df_out = df_out.loc[index,]
    df_out = df_out.reindex()
    df_summary = df_out.describe()
    df_out.to_csv("19simulation_N200_corrected_variance1_robust.csv",index=False)
    print("--- %s seconds ---" % (time.time() - start_time))
