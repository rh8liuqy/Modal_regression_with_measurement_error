import numpy as np
import pandas as pd
from scipy.special import loggamma
from scipy.special import digamma
from scipy.stats import f as fdist
from scipy.optimize import minimize
import numdifftools as nd
import multiprocessing as mp
import time

# the log transformed probability density function(pdf) of beta distribution with mean = a/(a+b).
# x: the observed values.
# a: a shape parameter.
# b: a shape parameter.
#output: the log transformed pdf.
def beta_lpdf(x,a,b):
    p1 = -loggamma(a)-loggamma(b)+loggamma(a+b)
    p2 = (a-1.0)*np.log(x)+(b-1.0)*np.log(1.0-x)
    output = p1+p2
    return output

# log transformed probability density function(pdf) of beta distribution in the regression setting.
# params[0]: the log transformed parameter m.
# params[1:len(params)]: betas.
# output: the log transformed pdf in the regression setting.
def beta_reg_lpdf(params,y,X):
    logm = params[0]
    m = np.exp(logm)
    betas = params[1:len(params)]
    betas = np.array(betas)
    etaX = np.matmul(X,betas)
    thetaX = 1.0/(1.0+np.exp(-etaX))
    output = beta_lpdf(y, 1.0+m*thetaX, 1.0+m*(1.0-thetaX))
    return output

# the MLE of the naive beta regression model.
# inits: the initial values unknown parameters.
# y: the observed dependent variables.
# X: the design matrix.
# output: the MLE estimation of the unknown parameters.
def beta_reg_MLE(inits,y,X):
    inits = np.array(inits)
    def neglogll(params):
        output = beta_reg_lpdf(params,y,X)
        return -2*np.sum(output)
    point_est = minimize(neglogll, 
                      x0=inits, 
                      method = 'Nelder-Mead')
    if point_est['success']:
        point_est = point_est['x']
    else:
        point_est = np.repeat(np.nan,len(point_est['x']))
    output = point_est
    return output

# the standard deviation of the naive MLE.
# params: the MLE estimation of unknown parameters.
# y: the observed dependent variables.
# X: the design matrix.
# output: the standard deviation associated with the naive MLE.
def beta_reg_sd(params,y,X):
    p = params.shape[0]
    def loglikelihood(params):
        output = -np.sum(beta_reg_lpdf(params,y,X))
        return output
    hessfunc = nd.Hessian(loglikelihood)
    try:
        output = np.linalg.inv(hessfunc(params))
    except:
        output = np.linalg.inv(hessfunc(params)+np.identity(p)*1e-6)
    output = np.sqrt(np.diag(output))
    return output

# the random number generator of random vector T in Stefanski et al. (2005). See Equation (7) in the main article.
# monte_carlo_size: the size of the random vector T. In Equation (7), this is denoted as B.
# sample_size: the sample size, which is as the same as the size of dependent variable y.
# repeated_measure: the number of the repeated measures.
# output: the generated random vector T.
def RV_T(monte_carlo_size,sample_size,repeated_measure=3):
    output = np.zeros(monte_carlo_size*sample_size)
    output = output.reshape(monte_carlo_size,sample_size)
    for i in range(sample_size):
        tZ = np.random.normal(0,1,monte_carlo_size*(repeated_measure-1))
        tZ1 = tZ[0:monte_carlo_size]
        mones = np.ones(repeated_measure-1)
        mones = mones.reshape(repeated_measure-1,1)
        dominator = (tZ.reshape(repeated_measure-1,monte_carlo_size).T)**2
        dominator = dominator @ mones
        dominator = dominator.flatten()
        T1 = tZ1/np.sqrt(dominator)
        output[:,i] = T1
    output = output.T
    output = np.concatenate(output)
    return output

# the corrected log-pdf of the beta regression model.
# params: all the unknown parameters of the beta regression model.
# W: the error contaminated covariates.
# Z: the error free covariates.
# nj: the number of repeated measures. See Equation (7) in the main article.
# S: the sample standard deviation of $\widetilde{W}_j=\left\{W_{j, k}\right\}_{k=1}^{n_j}$. See Equation (6) in the main article.
# B: the value of B in Equation (7).
# T: the generated random vector T.
# output: the corrected log-pdf of the beta regression model.
def correct_beta_reg_lpdf(params,y,W,Z,nj,S,B,T):
    logm = params[0]
    m = np.exp(logm)
    betas = params[1:len(params)]
    n = y.shape[0]
    y = np.repeat(y,B)
    W = np.repeat(W,B,0)
    Z = np.repeat(Z,B,0)
    S = np.repeat(S,B,0)
    W_imag = np.sqrt((nj-1)/nj)*S*T
    W = W + 0j
    W.imag = W_imag
    WZ = np.column_stack([W,Z])
    etaW = np.matmul(WZ,betas)
    thetaW = 1.0/(1.0+np.exp(-etaW))
    output = beta_lpdf(y, 1.0+m*thetaW, 1.0+m*(1.0-thetaW))
    output = np.real(output)
    output = output.reshape(n,B)
    output = np.mean(output,axis=1)
    return output

# the corrected score of each observation.
# y: the observed dependent variable. 
# W: the error contaminated covariates.
# Z: the error free covariates.
# nj: the number of repeated measures. See Equation (7) in the main article.
# S: the sample standard deviation of $\widetilde{W}_j=\left\{W_{j, k}\right\}_{k=1}^{n_j}$. See Equation (6) in the main article.
# B: the value of B in Equation (7).
# T: the generated random vector T.
# output: the corrected score of each observation.
def correct_score_n(params,y,W,Z,nj,S,B,T):
    logm = params[0]
    m = np.exp(logm)
    betas = params[1:len(params)]
    n = y.shape[0]
    y = np.repeat(y,B)
    W = np.repeat(W,B,0)
    Z = np.repeat(Z,B,0)
    S = np.repeat(S,B,0)
    W_imag = np.sqrt((nj-1)/nj)*S*T
    W = W + 0j
    W.imag = W_imag
    WZ = np.column_stack([W,Z])
    etaW = np.matmul(WZ,betas)
    thetaW = 1.0/(1.0+np.exp(-etaW))
    output = np.zeros((len(params),n))
    pm1 = digamma(2+m)
    pm2 = thetaW*digamma(1+m*thetaW)
    pm3 = (1-thetaW)*digamma(1+m*(1-thetaW))
    pm4 = thetaW*np.log(y)+(1-thetaW)*np.log(1-y)
    psim = pm1-pm2-pm3+pm4
    psim = np.real(psim)
    psim = psim.reshape(n,B)
    psim = np.mean(psim,axis=1)
    output[0,:] = psim
    pb1 = -m*digamma(1+m*thetaW)
    pb2 = m*digamma(1+m*(1-thetaW))
    pb3 = m*np.log(y/(1-y))
    gprime = np.exp(-etaW)/(1+np.exp(-etaW))**2
    for i in range(len(betas)):
        psib = (pb1+pb2+pb3)*gprime*WZ[:,i]
        psib = np.real(psib)
        psib = psib.reshape(n,B)
        psib = np.mean(psib,axis=1)
        output[i+1,:] = psib
    return output

# the summation of corrected score of all observations.
# y: the observed dependent variable. 
# W: the error contaminated covariates.
# Z: the error free covariates.
# nj: the number of repeated measures. See Equation (7) in the main article.
# S: the sample standard deviation of $\widetilde{W}_j=\left\{W_{j, k}\right\}_{k=1}^{n_j}$. See Equation (6) in the main article.
# B: the value of B in Equation (7).
# T: the generated random vector T.
# output: the summation of corrected score of all observations.
def correct_score(params,y,W,Z,nj,S,B,T):
    output = correct_score_n(params,y,W,Z,nj,S,B,T)
    output = np.sum(output,axis=1)
    return output

# the M estimation of all unknown parameters (point esitmation only).
# inits: the initial values unknown parameters.
# y: the observed dependent variable. 
# W: the error contaminated covariates.
# Z: the error free covariates.
# nj: the number of repeated measures. See Equation (7) in the main article.
# S: the sample standard deviation of $\widetilde{W}_j=\left\{W_{j, k}\right\}_{k=1}^{n_j}$. See Equation (6) in the main article.
# B: the value of B in Equation (7).
# T: the generated random vector T.
# output: the M estimation (point esitmation only) of all unknown parameters.
def correct_beta_reg_point_est(inits,y,W,Z,nj,S,B,T):
    inits = np.array(inits)
    def neglogll(params):
        output = correct_beta_reg_lpdf(params,y,W,Z,nj,S,B,T)
        return -2*np.sum(output)
    point_est = minimize(neglogll, 
                         x0=inits, 
                         method = 'Nelder-Mead')
    if point_est['success']:
        point_est = point_est['x']
    else:
        point_est = np.repeat(np.nan,len(point_est['x']))
    output = point_est
    return output

# the M estimation of all unknown parameters (point esitmation and standard deviation associated with it).
# inits: the initial values unknown parameters.
# y: the observed dependent variable. 
# W: the error contaminated covariates
# Z: the error free covariates.
# nj: the number of repeated measures. See Equation (7) in the main article.
# S: the sample standard deviation of $\widetilde{W}_j=\left\{W_{j, k}\right\}_{k=1}^{n_j}$. See Equation (6) in the main article.
# B: the value of B in Equation (7).
# T: the generated random vector T.
# output: the M estimation (point esitmation and standard deviation associated with it) of all unknown parameters.
def correct_beta_reg_inference(inits,y,W,Z,nj,S,B,T):
    inits = np.array(inits)
    def neglogll(params):
        output = correct_beta_reg_lpdf(params,y,W,Z,nj,S,B,T)
        return -2*np.sum(output)
    point_est = minimize(neglogll, 
                         x0=inits, 
                         method = 'Nelder-Mead')
    if point_est['success']:
        point_est = point_est['x']
    else:
        point_est = np.repeat(np.nan,len(point_est['x']))
    score_n = correct_score_n(point_est,y,W,Z,nj,S,B,T)
    A = nd.Jacobian(lambda x: correct_score(x,y,W,Z,nj,S,B,T))
    Amat = A(point_est)
    AS = np.linalg.solve(Amat,score_n)
    params_cov = np.sqrt(np.diag(np.matmul(AS,AS.T)))
    output = np.concatenate((point_est,params_cov))
    return output

# the hotelling T statistics. See Equation (13).
# params: the point estimation of all unknown parameters.
# y: the observed dependent variable. 
# W: the error contaminated covariates.
# Z: the error free covariates.
# nj: the number of repeated measures. See Equation (7) in the main article.
# S: the sample standard deviation of $\widetilde{W}_j=\left\{W_{j, k}\right\}_{k=1}^{n_j}$. See Equation (6) in the main article.
# B: the value of B in Equation (7).
# T: the generated random vector T.
# output: the hotelling T statistics.
def hotelling_T(params,y,W,Z,nj,S,B,T):
    logm = params[0]
    m = np.exp(logm)
    n = y.shape[0]
    p =len(params)
    betas = np.array(params[1:p])
    s = np.zeros((2,n))
    y = np.repeat(y,B)
    Z = np.repeat(Z,B,0)
    W = np.repeat(W,B,0)
    S = np.repeat(S,B,0)
    W_complex = W + 0j
    W_complex.imag = np.sqrt((nj-1)/nj)*S*T
    WZ = np.column_stack([W_complex,Z])
    etaW = np.matmul(WZ, betas)
    thetaW = 1/(1+np.exp(-etaW))
    s11 = np.log(y)
    s12 = -digamma(1+m*thetaW)
    s13 = digamma(2+m)
    s1 = s11+s12+s13
    s1 = np.real(s1)/B
    s1 = s1.reshape(n,B)
    s1 = np.sum(s1,axis=1)
    s21 = y*np.log(y)
    s22 = -((1+m*thetaW)*(digamma(2+m*thetaW)-digamma(3+m)))/(2+m)
    s2 = s21+s22
    s2 = np.real(s2)/B
    s2 = s2.reshape(n,B)
    s2 = np.sum(s2,axis=1)
    s[0,:] = s1
    s[1,:] = s2
    sbar = np.sum(s,axis=1)/n
    sbar = sbar.reshape(2,1)
    sdiff = s - sbar
    sigmahat = 1/(n*(n-1))*np.matmul(sdiff,sdiff.T)
    Tstat = (n-2)/(2*(n-1))*sbar.T@np.linalg.inv(sigmahat)@sbar
    Tstat = Tstat.flatten()[0]
    pvalue = 1-fdist.cdf(Tstat,2,n-2)
    output = {'T^2_statistic':Tstat,'p_value':pvalue}
    return output

# the bootstrap procedure of the hotelling T statistics.
# params: the point estimation of all unknown parameters.
# y: the observed dependent variable. 
# W: the error contaminated covariates.
# Z: the error free covariates.
# nj: the number of repeated measures. See Equation (7) in the main article.
# S: the sample standard deviation of $\widetilde{W}_j=\left\{W_{j, k}\right\}_{k=1}^{n_j}$. See Equation (6) in the main article.
# B: the value of B in Equation (7).
# T: the generated random vector T.
# bootstrap_B: the number of bootstrap.
# output: the p-value associated with the bootstrap procedure of hotelling T statistics.
def hotelling_bootstrap(params,y,W,Z,nj,S,B,T,bootstrap_B):
    if np.isnan(params).any():
        output = {'p-value':np.nan}
        return output
    ##calculation of xt
    n = len(y)
    Wbar = np.mean(W)
    SWi = (S)**2
    SW = np.cov(W)
    sigmaui = SWi / nj
    sigmau = np.mean(sigmaui)
    sigmax = SW - sigmau
    khat = (sigmax**(1/2))*(SW**(-1/2))
    xti = Wbar + khat*(W-Wbar)
    designx = np.column_stack((xti,Z))
    ##original M estimation
    betahat = params[1:len(params)]
    mhat = np.exp(params[0])
    Torigin = hotelling_T(params,y,W,Z,nj,S,B,T)['T^2_statistic']
    Tb = np.zeros(bootstrap_B)
    ##protect original y and w
    yb = np.copy(y)
    b = 0
    br = 0
    while b < bootstrap_B:
        for i in range(n):
            etaxi = np.matmul(designx[i,:], betahat)
            thetaxi = 1/(1+np.exp(-etaxi))
            yi = np.random.beta(1+mhat*thetaxi,1+mhat*(1-thetaxi))
            yb[i] = yi
        ## standard parameter estimation
        ## generate w 
        Wi = xti + np.random.normal(loc=0,scale=np.sqrt(sigmaui[i]),size=n)
        est_output= correct_beta_reg_point_est(params,yb,Wi,Z,nj,S,B,T)
        if br > 1000 + bootstrap_B:
            print("Maximum number of bootstrap iterations reached!")
            break
        if not(np.isnan(est_output).any()):
            omegahatb = est_output
            Tb[b] = hotelling_T(omegahatb,yb,Wi,Z,nj,S,B,T)['T^2_statistic']
            b = b + 1
            br = br + 1
        else:
            b = b
            br = br + 1
    output = {'p-value':np.mean(Tb > Torigin)}
    if br > 1000 + bootstrap_B:
        output = {'p-value':np.nan}
    return output
