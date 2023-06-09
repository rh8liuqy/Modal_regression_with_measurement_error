from Beta_reg import *
from scipy.stats import norm

def m1(sample_size,SigmaW=np.sqrt(1.2)):
    beta0 = 0.25
    beta1 = 0.25
    beta2 = 0.25
    m = 3.0
    repeated_measure=3
    output = np.zeros(sample_size*4)
    output = output.reshape(sample_size,4)
    for i in range(sample_size):
        Z1 = np.random.binomial(1,0.5,1)
        X1 = np.random.normal(((Z1 == 1)+0)-((Z1 == 0)),1,size=1)
        etaX = beta0 + beta1*(X1) + beta2*Z1
        thetaX = 1/(1+np.exp(-etaX))
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

def m2(sample_size):
    beta0 = 1
    beta1 = 1
    beta2 = 1
    beta3 = 1
    m = 40
    SigmaW=np.sqrt(1.2)
    output = np.zeros(sample_size*4)
    output = output.reshape(sample_size,4)
    for i in range(sample_size):
        Z1 = np.random.binomial(1,0.5,1)
        X1 = np.random.normal(((Z1 == 1)+0)-((Z1 == 0)),1,size=1)
        etaX = beta0 + beta1*(X1) + beta2*Z1 + beta3*(X1)**2
        thetaX = 1/(1+np.exp(-etaX))
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

def m3(sample_size):
    beta0 = 1
    beta1 = 1
    beta2 = 1
    m = 3
    SigmaW=np.sqrt(1.2)
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

def m4(sample_size):
    beta0 = 1
    beta1 = 1
    beta2 = 1
    m = 3
    SigmaW=np.sqrt(1.2)
    output = np.zeros(sample_size*4)
    output = output.reshape(sample_size,4)
    i = 0
    while i < sample_size:
        Z1 = np.random.binomial(1,0.5,1)
        X1 = np.random.normal(((Z1 == 1)+0)-((Z1 == 0)),1,size=1)
        etaX = beta0 + beta1*(X1) + beta2*Z1
        thetaX = 1/(1+np.exp(-etaX))
        if thetaX < 0.5:
            Y = np.random.gumbel(loc=thetaX,scale=(1-2*thetaX)*(np.euler_gamma*(2+m)),size=1)
            U1 = np.random.normal(0,SigmaW,1)
            U2 = np.random.normal(0,SigmaW,1)
            U3 = np.random.normal(0,SigmaW,1)
            W1 = X1 + U1
            W2 = X1 + U2
            W3 = X1 + U3
            SigmaWhat = np.std([W1,W2,W3],ddof=1)
            Wbar = np.mean([W1,W2,W3])
            output[i,] = np.column_stack((Y,Wbar,SigmaWhat,Z1))[0]
            i = i + 1
        else:
            i = i
    output = pd.DataFrame(output, columns=['Y','Wbar','SigmaWhat','Z1'])
    output['Y'] = output['Y']-np.min(output['Y'])
    output['Y'] = output['Y']/np.max(output['Y'])
    index = output['Y'] == 0
    output.loc[index,'Y'] = 1e-6
    index = output['Y'] == 1
    output.loc[index,'Y'] = 1-1e-6
    return output

def Table1(seed,method,n,SigmaW,B):
    np.random.seed(seed)
    condition = True
    while condition:
        ## data simulation
        df1 = m1(sample_size=n,SigmaW=SigmaW)
        y = df1['Y'].to_numpy()
        W = df1['Wbar'].to_numpy()
        Z = df1['Z1'].to_numpy()
        Z = np.column_stack([np.ones(Z.shape[0]),Z])
        sigmaW = df1['SigmaWhat'].to_numpy()
        if method == "MCCL":
            T = RV_T(monte_carlo_size=B,sample_size=n,repeated_measure=3)
            output = correct_beta_reg_point_est([np.log(3),0.25,0.25,0.25],
                                                y=y,W=W,Z=Z,nj=3,S=sigmaW,
                                                B=B,T=T)
        else:
            X = np.column_stack([W,Z])
            output = beta_reg_MLE([np.log(3),0.25,0.25,0.25],y,X)
        condition = np.isnan(output).any() or (output > 10.0).any()
    return output

def simu_Table1(simu_size,filename,method,n,SigmaW,B):
    simu_args = [None]*simu_size
    for i in range(simu_size):
        simu_args[i] = [i,method,n,SigmaW,B]
    start_time = time.time()
    with mp.Pool(24) as p:
        output = p.starmap(Table1, simu_args)
    df_out = pd.DataFrame(output,
                          columns=["logm","b1","b0","b2"])
    df_summary = print(np.round(df_out.describe(),2))
    df_out.to_csv(filename,index=False)
    print("--- %s seconds ---" % (time.time() - start_time))
    return None

def Figure1(indepedent,method,seed,n,SigmaW,B):
    np.random.seed(seed)
    condition = True
    while condition:
        ## data simulation
        if not indepedent:
            df1 = m1(sample_size=n,SigmaW=SigmaW)
            y = df1['Y'].to_numpy()
            W = df1['Wbar'].to_numpy()
            Z = df1['Z1'].to_numpy()
            Z = np.column_stack([np.ones(Z.shape[0]),Z])
            sigmaW = df1['SigmaWhat'].to_numpy()
        else:
            beta0 = 0.25
            beta1 = 0.25
            beta2 = 0.25
            m = 3.0
            repeated_measure=3
            output = np.zeros(n*4)
            output = output.reshape(n,4)
            for i in range(n):
                Z1 = np.random.binomial(1,0.5,1)
                X1 = np.random.normal(0,1,size=1)
                etaX = beta0 + beta1*(X1) + beta2*Z1
                thetaX = 1/(1+np.exp(-etaX))
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
            y = output['Y'].to_numpy()
            W = output['Wbar'].to_numpy()
            Z = output['Z1'].to_numpy()
            Z = np.column_stack([np.ones(Z.shape[0]),Z])
            sigmaW = output['SigmaWhat'].to_numpy()
        if method == "MCCL":
            T = RV_T(monte_carlo_size=B,sample_size=n,repeated_measure=3)
            output = correct_beta_reg_point_est([np.log(3),0.25,0.25,0.25],
                                                y=y,W=W,Z=Z,nj=3,
                                                S=sigmaW,B=B,T=T)
        else:
            X = np.column_stack([W,Z])
            output = beta_reg_MLE([np.log(3),0.25,0.25,0.25],y,X)
        condition = np.isnan(output).any() or (output > 10.0).any()
    return output

def simu_Figure1(simu_size,filename,indepedent,method,n,SigmaW,B):
    simu_args = [None]*simu_size
    for i in range(simu_size):
        simu_args[i] = [indepedent,method,i,n,SigmaW,B]
    start_time = time.time()
    with mp.Pool(24) as p:
        output = p.starmap(Figure1, simu_args)
    df_out = pd.DataFrame(output,columns=["logm","b1","b0","b2"])
    df_summary = print(np.round(df_out.describe(),2))
    df_out.to_csv(filename,index=False)
    print("--- %s seconds ---" % (time.time() - start_time))
    return None

def Table2(method,seed,n,SigmaW,B):
    np.random.seed(seed)
    condition = True
    while condition:
        df1 = m1(sample_size=n,SigmaW=SigmaW)
        y = df1['Y'].to_numpy()
        W = df1['Wbar'].to_numpy()
        Z = df1['Z1'].to_numpy()
        Z = np.column_stack([np.ones(Z.shape[0]),Z])
        sigmaW = df1['SigmaWhat'].to_numpy()
        if method == "MCCL":
            T = RV_T(monte_carlo_size=B,sample_size=n,repeated_measure=3)
            output = correct_beta_reg_inference([np.log(3),0.25,0.25,0.25],
                                                y=y,W=W,Z=Z,nj=3,
                                                S=sigmaW,B=B,T=T)
        else:
            X = np.column_stack([W,Z])
            point_est = beta_reg_MLE([np.log(3),0.25,0.25,0.25],y,X)
            cov_est = beta_reg_sd(params=point_est,y=y,X=X)
            output = np.concatenate((point_est,cov_est))
        condition = np.isnan(output).any() or (output > 10.0).any()
    return output

def simu_Table2(simu_size,filename,method,n,SigmaW,B):
    simu_args = [None]*simu_size
    for i in range(simu_size):
        simu_args[i] = [method,i,n,SigmaW,B]
    start_time = time.time()
    with mp.Pool(24) as p:
        output = p.starmap(Table2, simu_args)
    df_out = pd.DataFrame(output,columns=["logm","b1","b0","b2",
                                          "logm_se","b1_se","b0_se","b2_se"])
    df_summary = print(np.round(df_out.describe(),2))
    df_out.to_csv(filename,index=False)
    print("--- %s seconds ---" % (time.time() - start_time))
    return None

def Table3(method,known_variance,seed,n,B):    
    np.random.seed(seed)
    condition = True
    while condition:
        beta0 = 0.25
        beta1 = 0.25
        beta2 = 0.25
        m = 3.0
        repeated_measure=3
        output = np.zeros(n*4)
        output = output.reshape(n,4)
        for i in range(n):
            Z1 = np.random.binomial(1,0.5,1)
            X1 = np.random.normal(((Z1 == 1)+0)-((Z1 == 0)),1,size=1)
            etaX = beta0 + beta1*(X1) + beta2*Z1
            thetaX = 1/(1+np.exp(-etaX))
            Y = np.random.beta(1+m*thetaX,1+m*(1-thetaX),size=1)
            U1 = np.random.laplace(0,np.sqrt(1/2),1)
            U2 = np.random.laplace(0,np.sqrt(1/2),1)
            U3 = np.random.laplace(0,np.sqrt(1/2),1)
            W1 = X1 + U1
            W2 = X1 + U2
            W3 = X1 + U3
            SigmaWhat = np.std([W1,W2,W3],ddof=1)
            Wbar = np.mean([W1,W2,W3])
            output[i,] = np.column_stack((Y,Wbar,SigmaWhat,Z1))[0]
        df1 = pd.DataFrame(output, columns=['Y','Wbar','SigmaWhat','Z1'])
        y = df1['Y'].to_numpy()
        W = df1['Wbar'].to_numpy()
        Z = df1['Z1'].to_numpy()
        Z = np.column_stack([np.ones(Z.shape[0]),Z])
        if known_variance:
            sigmaW = np.repeat(1,n)
        else:
            sigmaW = df1['SigmaWhat'].to_numpy()
        if method == "MCCL":
            T = RV_T(monte_carlo_size=B,sample_size=n,repeated_measure=3)
            output = correct_beta_reg_point_est([np.log(3),0.25,0.25,0.25],
                                                y=y,W=W,Z=Z,nj=3,
                                                S=sigmaW,B=B,T=T)
        else:
            X = np.column_stack([W,Z])
            output = beta_reg_MLE([np.log(3),0.25,0.25,0.25],y,X)
        condition = np.isnan(output).any() or (output > 10.0).any()
    return output

def simu_Table3(simu_size,filename,known_variance,method,n,B):
    simu_args = [None]*simu_size
    for i in range(simu_size):
        simu_args[i] = [method,known_variance,i,n,B]
    start_time = time.time()
    with mp.Pool(24) as p:
        output = p.starmap(Table3, simu_args)
    df_out = pd.DataFrame(output,columns=["logm","b1","b0","b2"])
    df_summary = print(np.round(df_out.describe(),2))
    df_out.to_csv(filename,index=False)
    print("--- %s seconds ---" % (time.time() - start_time))
    return None

def Figure2(seed,n,SigmaW,B):
    np.random.seed(seed)
    condition = True
    while condition:
        ## data simulation
        df1 = m1(sample_size=n,SigmaW=SigmaW)
        y = df1['Y'].to_numpy()
        W = df1['Wbar'].to_numpy()
        Z = df1['Z1'].to_numpy()
        Z = np.column_stack([np.ones(Z.shape[0]),Z])
        sigmaW = df1['SigmaWhat'].to_numpy()
        T = RV_T(monte_carlo_size=B,sample_size=n,repeated_measure=3)
        point_est = correct_beta_reg_point_est([np.log(3),0.25,0.25,0.25],
                                               y=y,W=W,Z=Z,nj=3,S=sigmaW,
                                               B=B,T=T)
        if (not np.isnan(point_est).any()) and (point_est < 10.0).any():
            output = hotelling_bootstrap(params=point_est,
                                         y=y,
                                         W=W,
                                         Z=Z,
                                         nj=3,
                                         S=sigmaW,
                                         B=B,
                                         T=T,
                                         bootstrap_B=300)['p-value']
            condition = np.isnan(output)
    return output

def simu_Figure2(seed_start,seed_end,filename,n,SigmaW,B):
    simu_size = seed_end - seed_start + 1
    simu_args = [None]*simu_size
    seeds = np.arange(seed_start,seed_end+1,1)
    for i in range(simu_size):
        simu_args[i] = [seeds[i],n,SigmaW,B]
    start_time = time.time()
    with mp.Pool(24) as p:
        output = p.starmap(Figure2, simu_args)
    df_out = pd.DataFrame(output,columns=["p-value"])
    df_out.to_csv(filename,index=False)
    print("--- %s seconds ---" % (time.time() - start_time))
    return None

def Table4(model,seed,n,B):
    np.random.seed(seed)
    condition = True
    while condition:
        if model == "m2":
            df1 = m2(sample_size=n)
            inits = [ 2.3791916,-0.21304919,1.66358015,31.09859374]
        elif model == "m3":
            df1 = m3(sample_size=n)
            inits = [1.1674031,1.12631324,1.12523142,1.40476759]
        elif model == "m4":
            df1 = m4(sample_size=n)
            inits = [ 1.02481938,0.39741639,-0.74717532,-0.48585771]
        y = df1['Y'].to_numpy()
        W = df1['Wbar'].to_numpy()
        Z = df1['Z1'].to_numpy()
        Z = np.column_stack([np.ones(Z.shape[0]),Z])
        sigmaW = df1['SigmaWhat'].to_numpy()
        T = RV_T(monte_carlo_size=B,sample_size=n,repeated_measure=3)
        point_est = correct_beta_reg_point_est(inits,
                                               y=y,W=W,Z=Z,nj=3,S=sigmaW,
                                               B=B,T=T)
        if (not np.isnan(point_est).any()):
            output = hotelling_bootstrap(params=point_est,
                                         y=y,
                                         W=W,
                                         Z=Z,
                                         nj=3,
                                         S=sigmaW,
                                         B=B,
                                         T=T,
                                         bootstrap_B=300)['p-value']
            condition = np.isnan(output)
    return output

def simu_Table4(seed_start,seed_end,filename,model,n,B):
    simu_size = seed_end - seed_start + 1
    simu_args = [None]*simu_size
    seeds = np.arange(seed_start,seed_end+1,1)
    for i in range(simu_size):
        simu_args[i] = [model,seeds[i],n,B]
    start_time = time.time()
    with mp.Pool(24) as p:
        output = p.starmap(Table4, simu_args)
    df_out = pd.DataFrame(output,columns=["p-value"])
    df_out.to_csv(filename,index=False)
    print("--- %s seconds ---" % (time.time() - start_time))
    return None
