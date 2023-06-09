from Beta_reg import *
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from multiprocessing import Pool
from multiprocessing import cpu_count

def cbij_RNG(nij):
    z = np.random.normal(size=nij)
    zbar = np.mean(z)
    numerator = z - zbar
    denominator = np.sqrt(np.sum((z-zbar)**2))
    output = numerator/denominator
    return output

def wbi_RNG(zeta,W):
    nij = W.shape[0]
    wbar = np.mean(W)
    cbij = cbij_RNG(nij=nij)
    output = wbar + np.sqrt(zeta/nij)*np.sum(cbij*W)
    return output

def naive_estimation(inits,zeta,y,W,Z):
    n = y.shape[0]
    Wi = np.zeros(n)
    for i in range(n):
        Wi[i] = wbi_RNG(zeta,W[i,:])
    x = np.column_stack((Z,Wi))
    naive_est = beta_reg_MLE(inits=inits,y=y,X=x)
    output = naive_est
    return output

def SIMEX_simulation(inits,zeta,B,y,W,Z):
    p = inits.shape[0]
    output = np.zeros((B,p))
    i = 0
    condition = True
    while i < B:
        est = naive_estimation(inits=inits,
                               zeta=zeta,
                               y=y,
                               W=W,
                               Z=Z)
        if not(np.isnan(est).any()):
            output[i,] = est
            i = i + 1
    output = np.mean(output,axis=0)
    return output

def SIMEX_simulations(inits,zetas,B,y,W,Z):
    output = np.zeros((zetas.shape[0],inits.shape[0]))
    for i in range(zetas.shape[0]):
        output[i,] = SIMEX_simulation(inits=inits,
                                      zeta=zetas[i],
                                      B=B,
                                      y=y,
                                      W=W,
                                      Z=Z)
    return output

def SIMEX_extrapolation(seed,inits,zetas,B,y,W,Z):
    np.random.seed(seed)
    p = inits.shape[0]
    output = np.zeros(p)
    estimation = SIMEX_simulations(inits=inits,
                                   zetas=zetas,
                                   B=B,
                                   y=y,
                                   W=W,
                                   Z=Z)
    intercept = np.ones(zetas.shape[0])
    for i in range(p):
        y = estimation[:,i]
        designX = np.column_stack((intercept,zetas,zetas**2))
        reg = LinearRegression().fit(designX, y)
        predicted = reg.predict(np.array([[1,-1,1]]))
        output[i] = predicted
    return output

def SIMEX_bootstrap(num_bootstrap,inits,zetas,B,y,W,Z):
    arglists = [None]*num_bootstrap
    for i in range(num_bootstrap):
        yB,WB = resample(y,W)
        arglists[i] = [i,inits,zetas,B,yB,WB,Z]
    with Pool(cpu_count()) as p:
        output = p.starmap(SIMEX_extrapolation,arglists)
    return output

if __name__ == '__main__':
    np.random.seed(100)
    df1 = pd.read_csv("wishreg.csv")
    df1["ffq"] = df1["ffq"]/(8000)
    df1["fr1"] = (df1["fr1"] - np.mean(df1["fr1"]))/np.sqrt(np.var(df1["fr1"]))
    df1["fr2"] = (df1["fr2"] - np.mean(df1["fr2"]))/np.sqrt(np.var(df1["fr2"]))
    df1["fr3"] = (df1["fr3"] - np.mean(df1["fr3"]))/np.sqrt(np.var(df1["fr3"]))
    df1["fr4"] = (df1["fr4"] - np.mean(df1["fr4"]))/np.sqrt(np.var(df1["fr4"]))
    df1["fr5"] = (df1["fr5"] - np.mean(df1["fr5"]))/np.sqrt(np.var(df1["fr5"]))
    df1["fr6"] = (df1["fr6"] - np.mean(df1["fr6"]))/np.sqrt(np.var(df1["fr6"]))

    y = df1["ffq"].to_numpy()
    w = (df1["fr1"]+df1["fr2"]+df1["fr3"]+df1["fr4"]+df1["fr5"]+df1["fr6"])/6
    w = w.to_numpy()
    z = np.ones(w.shape[0])
    sigmaw = np.zeros(w.shape[0])
    for i in range(w.shape[0]):
        temp = df1.iloc[i,1:7].to_numpy()
        sigmaw[i] = np.std(temp,ddof=1)
    x = np.column_stack((z,w))

    B = 100
    T = RV_T(monte_carlo_size=B,sample_size=y.shape[0],repeated_measure=6)
    MCCL = correct_beta_reg_inference(inits=np.array([1.1,1.053,-1.569]),
                                      y=y,
                                      W=w,
                                      Z=z,
                                      nj=6,
                                      S=sigmaw,
                                      B=B,
                                      T=T)
    print("MCCL:")
    print(np.round(MCCL[[2,1,0,5,4,3]],3))
    naive_est = beta_reg_MLE(inits=np.array([1.1,-1.58,0.27]),y=y,X=x)
    print("Naive Estimation:")
    print(np.round(naive_est[[1,2,0]],3))
    print("Naive Estimation SE:")
    print(np.round(beta_reg_sd(params=naive_est,y=y,X=x)[[1,2,0]],3))
