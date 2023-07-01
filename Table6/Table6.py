import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numdifftools as nd
from scipy.stats import norm
np.random.seed(100)
torch.manual_seed(100)

if torch.cuda.is_available():
    device = 'cuda:0'

# the statistical inference program associated with Table 6.
# Omega: the unknown parameters.
def table6(Omega):
    ## Define the number of Monte Carlo replicates
    B = 1000

    ## data import
    df1 = pd.read_csv("../BJadni.csv")

    x1 = df1["ento.change"].to_numpy()
    x2 = df1["hipp.change"].to_numpy()
    z = np.ones(x1.shape[0])

    W = np.column_stack((z,x1,x2))
    W = torch.tensor(W,
                     dtype = torch.double,
                     device=device)

    ## Define Z matrix
    n = W.shape[0]
    p = W.shape[1]

    Z = torch.normal(mean=torch.zeros(n*p*B,dtype = torch.double,device=device),
                     std=torch.ones(n*p*B,dtype = torch.double,device=device))
    Z = Z.reshape((n*B,p))

    ## Define y, the dependent variable
    y = df1["y"].to_numpy()
    y = torch.tensor(y,
                     dtype = torch.double,
                     device=device)

    y = torch.repeat_interleave(y,
                                repeats=B,
                                dim=0)

    def W_func(W,B,Omega,Z):
        # W is a n*p matrix in CUDA-torch
        # B is the number of Monte Carlo replicates
        # Omega is a p*p matrix in CUDA-torch
        # Z is a (n*B)*p matrix in CUDA-torch
        W = torch.repeat_interleave(W,
                                    repeats=B,
                                    dim=0)
        Omega = torch.sqrt(Omega) ## only works for diagnoal Omega
        image_part = torch.matmul(Z,Omega)
        W_comp = torch.complex(W,image_part)
        return W_comp

    W_comp = W_func(W,B,Omega,Z)

    ## Define lgamma
    def lgamma(x):
        p1 = 0.5*(torch.log(torch.ones(1,device=device)*torch.pi*2.0)-torch.log(x))
        p2 = x*(-torch.ones(1,device=device) + torch.log(x+1.0/(12.0*x-1.0/(10.0*x))))
        return p1+p2

    ## Define log-likelihood
    def dbeta(y,theta,m):
        ## y are observations
        ## theta is the mode
        ## m is the scale parameter
        alpha1 = 1.0 + m*theta
        alpha2 = 1.0 + m*(1.0-theta)
        output = lgamma(alpha1+alpha2)-\
            lgamma(alpha1)-\
                lgamma(alpha2)+\
                    (alpha1-1.0)*torch.log(y)+\
                        (alpha2-1.0)*torch.log(1.0-y)
        return output

    ## Define objective function
    def obj(parms):
        parms = torch.tensor(parms,dtype=torch.double,device=device)
        alpha = parms[0]
        beta1 = parms[1]
        beta2 = parms[2]
        logm = parms[3]
        eta = alpha*W_comp[:,0]+beta1*W_comp[:,1]+beta2*W_comp[:,2]
        eta = torch.flatten(eta)
        theta = torch.exp(-torch.exp(-eta))
        output = dbeta(y,theta,torch.exp(logm))
        output = -2.0*torch.real(torch.mean(output)/B)
        output = output.detach().cpu().numpy()
        return output

    M_est = minimize(obj,
                     [-0.697,-0.125,-0.216,2.772],
                     method="Nelder-Mead")

    ## Define digamma function
    def digamma(x):
        p1 = 1/(x)
        p2 = 1/(2*x**2)
        p3 = 5/(4*3*2*x**3)
        p4 = 3/(2*4*3*2*x**4)
        p5 = 47/(48*5*4*3*2*x**5)
        return torch.log(1/(p1+p2+p3+p4+p5))

    ## Define score function for each observation

    def score_n(parms):
        parms = torch.tensor(parms,dtype=torch.double,device=device)
        alpha = parms[0]
        beta1 = parms[1]
        beta2 = parms[2]
        logm = parms[3]
        m = torch.exp(logm)
        eta = alpha*W_comp[:,0]+beta1*W_comp[:,1]+beta2*W_comp[:,2]
        eta = torch.flatten(eta)
        theta = torch.exp(-torch.exp(-eta))
        output = np.zeros(((p+1),n))
        ## score w.r.t log m
        psilogm = digamma(2.0+m)-\
            theta*digamma(1.0+m*theta)-\
                (1.0-theta)*digamma(1.0+m*(1.0-theta))+\
                    theta*torch.log(y)+\
                        (1.0-theta)*torch.log(1.0-y)
        psilogm = psilogm.reshape((n,B))
        psilogm = torch.mean(torch.real(psilogm),axis=1)
        output[3,:] = psilogm.detach().cpu().numpy()
        ## score w.r.t betas
        gprime = torch.exp(-torch.exp(-eta)-eta)
        psibetas = (-m*digamma(1.0+m*theta)+m*digamma(1.0+m*(1.0-theta))+\
                    m*(torch.log(y)-torch.log(1.0-y)))*gprime
        psibetas = psibetas.reshape((n*B,1))*W_comp
        psibeta0 = psibetas[:,0]
        psibeta0 = psibeta0.reshape((n,B))
        psibeta0 = torch.mean(torch.real(psibeta0),axis=1)
        psibeta1 = psibetas[:,1]
        psibeta1 = psibeta1.reshape((n,B))
        psibeta1 = torch.mean(torch.real(psibeta1),axis=1)
        psibeta2 = psibetas[:,2]
        psibeta2 = psibeta2.reshape((n,B))
        psibeta2 = torch.mean(torch.real(psibeta2),axis=1)
        output[0,:] = psibeta0.detach().cpu().numpy()
        output[1,:] = psibeta1.detach().cpu().numpy()
        output[2,:] = psibeta2.detach().cpu().numpy()
        return output

    def score(parms):
        output = score_n(parms)
        output = np.sum(output,axis=1)
        return output

    ## Define matrix A
    Amat_func = nd.Jacobian(score)
    Amat = Amat_func(M_est.x)/n
    Ainv = np.linalg.inv(Amat)

    ## Define score_n mat
    score_n_mat = score_n(M_est.x)

    ## Define matrix B
    Bmat = np.matmul(score_n_mat, score_n_mat.T)/n

    ## Define V matrix
    Vmat = Ainv @ Bmat @ Ainv.T
    se = np.sqrt(np.diag(Vmat)/n)

    if M_est.success:
        out_dict = {"alpha":M_est.x[0],
                    "beta1":M_est.x[1],
                    "beta2":M_est.x[2],
                    "log(M)":M_est.x[3],
                    "alpha_se":se[0],
                    "beta1_se":se[1],
                    "beta2_se":se[2],
                    "log(M)_se":se[3]}
    else:
        out_dict = {"alpha":np.nan,
                    "beta1":np.nan,
                    "beta2":np.nan,
                    "log(M)":np.nan,
                    "alpha_se":np.nan,
                    "beta1_se":np.nan,
                    "beta2_se":np.nan,
                    "log(M)_se":np.nan}

    df_tab = pd.DataFrame({"effect":["Intercept",
                                     "ERC.change",
                                     "HPC.change",
                                     "log(m)"],
                           "point.est":[out_dict["alpha"],
                                        out_dict["beta1"],
                                        out_dict["beta2"],
                                        out_dict["log(M)"]],
                           "SE":[out_dict["alpha_se"],
                                 out_dict["beta1_se"],
                                 out_dict["beta2_se"],
                                 out_dict["log(M)_se"]]})
    df_tab["p-value"] = 2.0*(1.0-norm.cdf(np.abs(df_tab["point.est"]/df_tab["SE"])))
    df_tab["95% Lower CL"] = df_tab["point.est"] - norm.ppf(0.975)*df_tab["SE"]
    df_tab["95% Upper CL"] = df_tab["point.est"] + norm.ppf(0.975)*df_tab["SE"]
    print(np.round(df_tab,4))
    return df_tab

## Naive Analysis
print("Naive Analysis")
Omega = torch.tensor([0.0,0.0,0.0],
                     dtype = torch.double,
                     device=device)
table6(torch.diag(Omega))

## ERC error-prone, variance = 0.15**2.0
print("ERC error-prone, variance = 0.15**2.0")
Omega = torch.tensor([0.0,0.15**2.0,0.0],
                     dtype = torch.double,
                     device=device)
table6(torch.diag(Omega))

## ERC error-prone, variance = 0.40**2.0
print("ERC error-prone, variance = 0.40**2.0")
Omega = torch.tensor([0.0,0.40**2.0,0.0],
                     dtype = torch.double,
                     device=device)
table6(torch.diag(Omega))

## HPC error-prone, variance = 0.15**2.0
print("HPC error-prone, variance = 0.15**2.0")
Omega = torch.tensor([0.0,0.0,0.15**2.0],
                     dtype = torch.double,
                     device=device)
table6(torch.diag(Omega))

## ERC and HPC error-prone, varianceERC = 0.15**2.0, varianceHPC = 0.15**2.0
print("ERC and HPC error-prone, varianceERC = 0.15**2.0, varianceHPC = 0.15**2.0")
Omega = torch.tensor([0.0,0.15**2.0,0.15**2.0],
                     dtype = torch.double,
                     device=device)
table6(torch.diag(Omega))

## ERC and HPC error-prone, varianceERC = 0.40**2.0, varianceHPC = 0.15**2.0
print("ERC and HPC error-prone, varianceERC = 0.40**2.0, varianceHPC = 0.15**2.0")
Omega = torch.tensor([0.0,0.40**2.0,0.15**2.0],
                     dtype = torch.double,
                     device=device)
table6(torch.diag(Omega))
