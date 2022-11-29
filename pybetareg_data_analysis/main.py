import numpy as np
import pandas as pd
from scipy.special import loggamma
from scipy.special import gamma
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import f as fdist
import numdifftools as nd

def jacobian(func,initial,delta=1e-3):
    f = func
    nrow = len(f(initial))
    ncol = len(initial)
    output = np.zeros(nrow*ncol)
    output = output.reshape(nrow,ncol)
    for i in range(nrow):
        for j in range(ncol):
            ej = np.zeros(ncol)
            ej[j] = 1
            dij = (f(initial+ delta * ej)[i] - f(initial- delta * ej)[i])/(2*delta)
            output[i,j] = dij
    return output

try:
    import cupy as cp
    def cu_digamma(x):
        x = cp.array(x)
        p1 = 1/(x)
        p2 = 1/(2*x**2)
        p3 = 5/(4*3*2*x**3)
        p4 = 3/(2*4*3*2*x**4)
        p5 = 47/(48*5*4*3*2*x**5)
        return cp.log(1/(p1+p2+p3+p4+p5))
    def cu_gamma(x):
        x = cp.array(x)
        p1 = cp.sqrt(2*cp.pi/x)
        p2 = (1/cp.exp(1)*(x+1/(12*x-1/(10*x))))**x
        return p1*p2
except Exception as e:
    print(e)
    print("cupy is not available.")

## fit() attributes
class fitoutput:
    def __init__(self,fitted):
        self.loglikelihood = fitted.fun / (-2)
        self.params_cov = fitted.params_cov
        self.message = fitted.message
        self.success = fitted.success
        self.column_names = fitted.column_names
        self.designmatrix = fitted.designmatrix
        self.y = fitted.y
        nparams = len(fitted.x)
        self.fitted = np.matmul(self.designmatrix, 
                                np.array(fitted.x)[1:nparams])
        self.fitted = 1/(1+np.exp(-self.fitted))
        self.residual = self.y - self.fitted
        if self.column_names == "NA":
            keysparams = ['log(m)']
            for i in range(nparams-1):keysparams.append('beta' + str(i))
            self.params = {keysparams[i]: 
                           fitted.x[i] for i in range(nparams)}
        else:
            keysparams = self.column_names
            keysparams = ['log(m)']
            for i in range(nparams-1):keysparams.append(self.column_names[i])
            self.params = {keysparams[i]: 
                           fitted.x[i] for i in range(nparams)}
    def summary(self):
        if self.column_names == "NA":print("Columns names are not given.")
        print("Success:"+str(self.success))
        print(self.message)
        print('"""')
        print("Beta Modal Regression Results".center(70,' '))
        print('='*70)
        print('coef'.rjust(20,' ' )+\
              'std err'.rjust(10,' ' )+\
              'z'.rjust(10,' ' )+\
              'P>|z|'.rjust(10,' ' )+\
              '[0.025'.rjust(10,' ' )+\
              '0.975]'.rjust(10,' ' ))
        print('-'*70)
        for i in range(len(self.params)):
            keyi = list(self.params.keys())[i]
            coefi = list(self.params.values())[i]
            stdi = np.sqrt(np.diag(self.params_cov))[i]
            zi = coefi/stdi
            pi = norm.cdf(-np.abs(zi)) * 2
            loweri = coefi - norm.ppf(1-0.05/2) * stdi
            upperi = coefi + norm.ppf(1-0.05/2) * stdi
            texti = keyi.ljust(10,' ')+\
                  "{:.4f}".format(coefi).rjust(10,' ')+\
                  "{:.3f}".format(stdi).rjust(10,' ')+\
                  "{:.3f}".format(zi).rjust(10,' ')+\
                  "{:.3f}".format(pi).rjust(10,' ')+\
                  "{:.3f}".format(loweri).rjust(10,' ')+\
                  "{:.3f}".format(upperi).rjust(10,' ')
            print(texti)
        print('='*70)
        print('"""')
        return
    def predict(self,x,transformed=False,alpha=0.05):
        x = np.array(x)
        betas = list(self.params.values())
        betas = np.array(betas)
        p = len(betas)
        betas = betas[1:p]
        pred = np.matmul(x, betas)
        if len(x.shape) > 1:
            std = np.sqrt(np.diag((x@self.params_cov[1:p,1:p]@x.T)))
        else:
            std = np.sqrt(x@self.params_cov[1:p,1:p]@x.T)
        lower = pred - norm.ppf(1-alpha/2) * std
        upper = pred + norm.ppf(1-alpha/2) * std
        if transformed == True:
            pred = 1/(1+np.exp(-pred))
            lower = 1/(1+np.exp(-lower))
            upper = 1/(1+np.exp(-upper))
        outdict = {'pred':pred.tolist(),
                   '['+str(np.round(alpha/2,3)):lower.tolist(),
                   str(np.round(1-alpha/2,3))+']':upper.tolist()}
        if len(x.shape) > 1:
            output = pd.DataFrame(outdict)
        else:
            output = pd.DataFrame(outdict,index=[0])
        outlength = len(str(output.round(4)))/(output.shape[0]+1)
        outlength = int(outlength)
        print('Prediction is successful.')
        print('Prediction in original scale:',transformed)
        print('='*outlength)
        print(output.round(4))
        print('='*outlength)
        return output

## beta modal without measurement errors
class betamodal:
    def __init__(self,x,y,initial="NA",column_names="NA",link="logit"):
        self.x = x
        self.y = y
        self.column_names = column_names
        self.link = link
        self.initial = initial
        print("Link function:"+self.link)
    def loglikelihood(self,params,echo=True):
        m = np.exp(params[0])
        betas = params[1:len(params)]
        if m < 0:
            if echo == True:
                print("m must be non-negative.")
            return np.nan
        if len(betas) != self.x.shape[1]:
            if echo == True:
                print("check betas values.")
            return np.nan
        betas = np.array(betas)
        etaX = np.matmul(self.x,betas)
        if self.link == "logit":
            thetaX = 1/(1+np.exp(-etaX))
        n = self.y.shape[0]
        p1 = n*loggamma(2+m)
        p2 = -np.sum(np.log(gamma(1+m*thetaX)*gamma(1+m*(1-thetaX))))
        p3 = m*np.sum(thetaX*np.log(self.y)+(1-thetaX)*np.log(1-self.y))
        return p1+p2+p3
    def hessinv(self,params):
        hessfunc = nd.Hessian(self.loglikelihood)
        try:
            output = np.linalg.inv(hessfunc(params,echo=False))
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            hessmat = hessfunc(params,echo=False)
            output = np.linalg.inv(hessmat+np.identity(params.shape[0])*1e-6)
        return output
    def fit(self):
        if self.initial == "NA":
            print("initial values are not assigned.")
            print("random initial values are assigned.")
            initialm = np.random.poisson(10,1)
            initialbeta = np.random.normal(0,1,size=self.x.shape[1])
            initial = np.concatenate((initialm,initialbeta))
        if self.initial != "NA":
            initial = np.array(self.initial)
        def neglogll(params):
            output = self.loglikelihood(params=params,echo=False)
            return -2*output
        output = minimize(neglogll, 
                          x0=initial.tolist(), 
                          method = 'Nelder-Mead')
        output['params_cov'] = -self.hessinv(output['x'])
        output['column_names'] = self.column_names
        output['designmatrix'] = self.x
        output['y'] = self.y
        return fitoutput(output)

## beta modal with measurement errors
### Random Vector T
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

###Digamma
def digamma(x):
    x = np.array(x)
    p1 = 1/(x)
    p2 = 1/(2*x**2)
    p3 = 5/(4*3*2*x**3)
    p4 = 3/(2*4*3*2*x**4)
    p5 = 47/(48*5*4*3*2*x**5)
    return np.log(1/(p1+p2+p3+p4+p5))

###Trigamma
def trigamma(x):
    x = np.array(x)
    p1 = 1/x
    p2 = 1/(2*x**2)
    p3 = 1/(6*x**3)
    p4 = -1/(30*x**5)
    p5 = 1/(42*x**7)
    p6 = -1/(30*x**9)
    return p1+p2+p3+p4+p5+p6

class fitoutput_measurement_error:
    def __init__(self,fitted):
        self.params_cov = fitted.params_cov
        self.message = fitted.message
        self.success = fitted.success
        self.column_names = fitted.column_names
        self.hotelling_T = fitted.hotelling_T
        nparams = len(fitted.x)
        if self.column_names == "NA":
            keysparams = ['log(m)']
            for i in range(nparams-1):keysparams.append('beta' + str(i))
            self.params = {keysparams[i]: 
                           fitted.x[i] for i in range(nparams)}
        else:
            keysparams = self.column_names
            keysparams = ['log(m)']
            for i in range(nparams-1):keysparams.append(self.column_names[i])
            self.params = {keysparams[i]: 
                           fitted.x[i] for i in range(nparams)}
    def summary(self):
        if self.column_names == "NA":print("Columns names are not given.")
        print("Success:"+str(self.success))
        print(self.message)
        print('"""')
        print("Beta Modal Regression Results With".center(70,' '))
        print("Measurement Error Adjustment".center(70,' '))
        print('='*70)
        print('coef'.rjust(20,' ' )+\
              'std err'.rjust(10,' ' )+\
              'z'.rjust(10,' ' )+\
              'P>|z|'.rjust(10,' ' )+\
              '[0.025'.rjust(10,' ' )+\
              '0.975]'.rjust(10,' ' ))
        print('-'*70)
        for i in range(len(self.params)):
            keyi = list(self.params.keys())[i]
            coefi = list(self.params.values())[i]
            stdi = np.sqrt(np.diag(self.params_cov))[i]
            zi = coefi/stdi
            pi = norm.cdf(-np.abs(zi)) * 2
            loweri = coefi - norm.ppf(1-0.05/2) * stdi
            upperi = coefi + norm.ppf(1-0.05/2) * stdi
            texti = keyi.ljust(10,' ')+\
                  "{:.4f}".format(coefi).rjust(10,' ')+\
                  "{:.3f}".format(stdi).rjust(10,' ')+\
                  "{:.3f}".format(zi).rjust(10,' ')+\
                  "{:.3f}".format(pi).rjust(10,' ')+\
                  "{:.3f}".format(loweri).rjust(10,' ')+\
                  "{:.3f}".format(upperi).rjust(10,' ')
            print(texti)
        print('='*70)
        print('"""')
        return

class betamodal_measurement_error:
    def __init__(self,y,w,z,sigmaw,initial,
                 monte_carlo_size=500,
                 repeated_measure=3,
                 column_names="NA",
                 link="logit",
                 CUDA = False):
        self.y = y
        self.w = w
        self.z = z
        self.sigmaw = sigmaw
        self.monte_carlo_size = monte_carlo_size
        self.repeated_measure = repeated_measure
        self.initial = initial
        self.column_names = column_names
        self.link = link
        self.CUDA = CUDA
        self.T = RV_T(self.monte_carlo_size,self.y.shape[0])

    def score_n(self,params):
        p = np.column_stack([self.w,self.z]).shape[1]
        repeated_measure = self.repeated_measure
        if self.CUDA == True:
            m = cp.array(cp.exp(params[0]))
            betas = cp.array(params[1:len(params)])
            output = cp.zeros((p+1)*len(self.y)).reshape(p+1,len(self.y))
            y = cp.array(self.y)
            z = cp.array(self.z)
            w = cp.array(self.w)
            sigmaw = cp.array(self.sigmaw)
            y = cp.repeat(y,self.monte_carlo_size)
            z = cp.repeat(z,self.monte_carlo_size,axis=0)
            w = cp.repeat(w,self.monte_carlo_size,axis=0)
            sigmaw = cp.repeat(sigmaw,self.monte_carlo_size)
            T = cp.array(self.T)
            w_complex = w + 0j
            repeated_measure = cp.array(repeated_measure)
            w_complex.imag = cp.sqrt((repeated_measure-1)/repeated_measure)*\
                sigmaw*T
            wz = cp.column_stack([w_complex,z])
            etaw = cp.matmul(wz, betas)
            thetaw = 1/(1+cp.exp(-etaw))
            pm1 = cu_digamma(2+m)
            pm2 = thetaw*cu_digamma(1+m*thetaw)
            pm3 = (1-thetaw)*cu_digamma(1+m*(1-thetaw))
            pm4 = thetaw*cp.log(y)+(1-thetaw)*cp.log(1-y)
            psim = pm1-pm2-pm3+pm4
            psim = cp.real(psim)/self.monte_carlo_size
            psim = psim.reshape(len(self.y),self.monte_carlo_size)
            psim = cp.sum(psim,axis=1)
            output[0,:] = psim
            pb1 = -m*cu_digamma(1+m*thetaw)
            pb2 = m*cu_digamma(1+m*(1-thetaw))
            pb3 = m*cp.log(y/(1-y))
            if self.link == "logit":
                gprime = cp.exp(-etaw)/(1+cp.exp(-etaw))**2
            for i in range(len(betas)):
                psib = (pb1+pb2+pb3)*gprime*wz[:,i]
                psib = cp.real(psib)/self.monte_carlo_size
                psib = psib.reshape(len(self.y),self.monte_carlo_size)
                psib = cp.sum(psib,axis=1)
                output[i+1,:] = psib
            output = cp.asnumpy(output)
        return output
    def score(self,params):
        output = self.score_n(params)
        output = np.sum(output,axis=1)
        return output
    def hotelling_T(self,params):
        if self.CUDA == True:
            output = cp.zeros(1)
            repeated_measure = self.repeated_measure
            m = cp.array(cp.exp(params[0]))
            n = len(self.y)
            betas = cp.array(params[1:len(params)],dtype=float)
            s = cp.zeros(2*len(self.y)).reshape(2,n)
            y = cp.repeat(cp.array(self.y),self.monte_carlo_size)
            z = cp.repeat(cp.array(self.z),self.monte_carlo_size,axis=0)
            w = cp.repeat(cp.array(self.w),self.monte_carlo_size,axis=0)
            sigmaw = cp.repeat(cp.array(self.sigmaw),self.monte_carlo_size)
            T = cp.array(self.T)
            w_complex = w + 0j
            w_complex.imag = cp.sqrt((repeated_measure-1)/repeated_measure)*\
                sigmaw*T
            wz = cp.column_stack([w_complex,z])
            etaw = cp.matmul(wz, betas)
            thetaw = 1/(1+cp.exp(-etaw))
            s11 = cp.log(y)
            s12 = -cu_digamma(1+m*thetaw)
            s13 = cu_digamma(2+m)
            s1 = s11+s12+s13
            s1 = cp.real(s1)/self.monte_carlo_size
            s1 = s1.reshape(n,self.monte_carlo_size)
            s1 = cp.sum(s1,axis=1)
            s21 = y*cp.log(y)
            s22 = -((1+m*thetaw)*(cu_digamma(2+m*thetaw)-cu_digamma(3+m)))/(2+m)
            s2 = s21+s22
            s2 = cp.real(s2)/self.monte_carlo_size
            s2 = s2.reshape(n,self.monte_carlo_size)
            s2 = cp.sum(s2,axis=1)
            s[0,:] = s1
            s[1,:] = s2
            sbar = cp.sum(s,axis=1)/n
            sbar = sbar.reshape(2,1)
            sdiff = s - sbar
            sigmahat = 1/(n*(n-1))*cp.matmul(sdiff,sdiff.T)
            output = (n-2)/(2*(n-1))*sbar.T@cp.linalg.inv(sigmahat)@sbar
            output = output.flatten()[0]
            output = cp.asnumpy(output)
        pvalue = 1-fdist.cdf(output,2,n-2)
        print("Hotelling's T^2 statistic and parametric bootstrap p-value.".\
              center(70,' '))
        print('='*70)
        print("Hotelling's T^2 statistic: {:.4f}".format(output))
        print("Asymptotic p-value: {:.4f}".format(pvalue))
        print('='*70)
        return {'T^2_statistic':float(output),'p_value':pvalue}
    def A_mat(self,params):
        output = jacobian(self.score, initial=params, delta=1e-6)
        output = output/len(self.y)
        return output
    def B_mat(self,params):
        psi = self.score_n(params)
        output = np.matmul(psi, psi.T)/len(self.y)
        return output
    def loglikelihood(self,params):
         repeated_measure = self.repeated_measure
         #n = self.y.shape[0]
         m = cp.array(cp.exp(params[0]))
         betas = cp.array(params[1:len(params)])
         y = cp.array(self.y)
         z = cp.array(self.z)
         w = cp.array(self.w)
         sigmaw = cp.array(self.sigmaw)
         y = cp.repeat(y,self.monte_carlo_size)
         z = cp.repeat(z,self.monte_carlo_size,axis=0)
         w = cp.repeat(w,self.monte_carlo_size,axis=0)
         sigmaw = cp.repeat(sigmaw,self.monte_carlo_size)
         T = cp.array(self.T)
         w_complex = w + 0j
         repeated_measure = cp.array(repeated_measure)
         w_complex.imag = cp.sqrt((repeated_measure-1)/repeated_measure)*\
             sigmaw*T
         wz = cp.column_stack([w_complex,z])
         etaw = cp.matmul(wz, betas)
         thetaw = 1/(1+cp.exp(-etaw))
         p1 = cp.log(cu_gamma(2+m))
         p2 = -cp.log(cu_gamma(1+m*thetaw)*cu_gamma(1+m*(1-thetaw)))
         p3 = m*(thetaw*cp.log(y)+(1-thetaw)*cp.log(1-y))
         output = p1+p2+p3
         output = output.reshape(len(self.y),self.monte_carlo_size)
         output = cp.real(output)/self.monte_carlo_size
         output = cp.sum(output,axis=1)
         output = cp.sum(output)
         return cp.asnumpy(output)
    def mle_est(self):
        def obj(params):
             output = self.loglikelihood(params)
             output = output*(-2)
             return output
        output = minimize(obj,
                          x0=self.initial.tolist(),
                          method='Nelder-Mead')
        return output

    def hotelling_bootstrap(self,bootstrap_B):
        ##calculation of xt
        n = len(self.y)
        wi = self.w
        wbar = np.mean(wi)
        swi = (self.sigmaw)**2
        ##sw need to be changed as vector version
        #sw = np.sum((wi-wbar)**2)/(n-1) to be changed as theortical covariance of W
        sw = np.cov(wi)
        sigmaui = swi / self.repeated_measure
        sigmau = np.mean(sigmaui)
        sigmax = sw - sigmau
        khat = (sigmax**(1/2))*(sw**(-1/2))
        xti = wbar + khat*(wi-wbar)
        designx = np.column_stack((xti,self.z))
        ##original MLE
        omegahat = np.copy(self.mle_est()['x'])
        betahat = omegahat[1:len(omegahat)]
        mhat = omegahat[0]
        Torigin = self.hotelling_T(omegahat)['T^2_statistic']
        Tb = np.zeros(bootstrap_B)
        ##protect original y and w
        yori = np.copy(self.y)
        wori = np.copy(self.w)
        yb = np.copy(self.y)
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
            wi = xti + np.random.normal(loc=0,scale=np.sqrt(sigmaui[i]),size=n)
            self.w = np.copy(wi)
            self.y = np.copy(yb)
            est_output= self.mle_est()
            if br > 1000 + bootstrap_B:
                print("Maximum number of bootstrap iterations reached!")
                break
            if est_output['success']:
                omegahatb = est_output['x']
                Tb[b] = self.hotelling_T(omegahatb)['T^2_statistic']
                b = b + 1
                br = br + 1
            else:
                b = b
                br = br + 1
        output = {'p-value':np.mean(Tb > Torigin)}
        if br > 1000 + bootstrap_B:
            output = {'p-value':np.nan}
        ##restore values of w and y
        self.w = wori
        self.y = yori
        return output
    
    def fit(self):
        output = self.mle_est()
        A = self.A_mat(output['x'])
        B = self.B_mat(output['x'])
        try:
            output['params_cov'] = np.linalg.inv(A)@B@np.linalg.inv(A).T/len(self.y)
        except Exception as e:
            print(e)
            Ainv = np.linalg.inv(A+np.identity(A.shape[0])*1e-6)
            output['params_cov'] = Ainv@B@Ainv.T/len(self.y)
        output['column_names'] = self.column_names
        output['hotelling_T'] = self.hotelling_T(output['x'])
        print('Model fitting completes'.center(70,'-'))
        return fitoutput_measurement_error(output)
