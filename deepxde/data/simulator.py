from math import *
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.stats import norm, chi2, beta
from stochastic.processes.continuous import BrownianMotion, PoissonProcess


class SimulatedSurvival:

    def __init__(self,T,step):
        self.T = T #100
        self.step = step #0.1
        self.t = np.arange(0,self.T,self.step)
        cdf = beta.cdf(self.t/self.T,8,1)
        pdf = beta.pdf(self.t/self.T,8,1)
        self.baseline = pdf/(1-cdf)
        #self.baseline = 0.01*(self.baseline+0.01)
            

    def generate_data(self, N, m=None, seed=0,test=False,trt=0):

        np.random.seed(seed)
        t = np.arange(0,self.T,self.step) #[0,100,0.1]
        alpha_list = []
        Z_list = []
        W_list = []
        Y_list = [] #observation time
        delta_list = []
        S_true_list = []

        for i in range(N):
            alpha = np.random.uniform(0,1,5)
            alpha_list.append(alpha)
            if test:
                Z = trt
            else:
                Z = np.random.binomial(1,0.5)  
            W = np.random.normal(0,1)
            Z_list.append(Z)
            W_list.append(W)
            X = alpha[0]+alpha[1]*np.sin(2*pi*t/self.T)+alpha[2]*np.cos(2*pi*t/self.T)+alpha[3]*np.sin(4*pi*t/self.T)+alpha[4]*np.cos(4*pi*t/self.T)
            X_cum = np.cumsum(X)*self.step
            XZ_cum = np.cumsum(X**2*Z)*self.step
            h = 0.05*np.exp(0.01*(X_cum+XZ_cum)+W+Z)
            #h = self.baseline * np.exp(0.1*(X_cum+XZ_cum)+W+Z)
            ch = np.cumsum(h)*self.step
            S = np.exp(-ch)
            S_true_list.append(S)
            u = np.random.uniform(0,1,1)[0]
            index = np.sum(s>u for s in S) #S[index-1]>u, S[index]<=u
            if index==len(t):
                fT = t[-1]
            elif index==0:
                fT = 0.0001
            else:
                fT = t[index-1]+self.step*(S[index-1]-u)/(S[index-1]-S[index])
            C = min(np.random.exponential(50,1)[0],self.T-1)
            Y_list.append(min(fT,C)) #observation time
            delta_list.append((fT<C)*1)

 
        print('Failue rate:', sum(delta_list)/len(delta_list))

        partition_y = np.arange(0,self.T,self.T/m) # fixed partition [1,100]
            
        y_n = len(partition_y)
        X = np.ones((N*y_n,m))
        y = np.ones((N*y_n,1))
        loc = np.ones((N*y_n,1))
        z = np.ones((N*y_n,1))
        w = np.ones((N*y_n,1))
        ind = np.ones((N*y_n,1))

        for i in range(N):
            Y = Y_list[i]
            Z = Z_list[i]
            W = W_list[i]
            delta = delta_list[i]
            alpha = alpha_list[i]
            prev_time = 0
            for j, time in enumerate(partition_y):
                loc[y_n*i+j] = [time]
                z[y_n*i+j] = Z
                w[y_n*i+j] = W
                if time < Y:
                    y[y_n*i+j] = [0]
                elif time >= Y:
                    y[y_n*i+j] = [delta]
                if prev_time <= Y:
                #if time <= Y:
                    ind[y_n*i+j] = [1]
                else:
                    ind[y_n*i+j] = [0]
                for k, now_t in enumerate(partition_y): 
                    X[y_n*i+j][k] = alpha[0]+alpha[1]*np.sin(2*pi*now_t/self.T)+alpha[2]*np.cos(2*pi*now_t/self.T)+alpha[3]*np.sin(4*pi*now_t/self.T)+alpha[4]*np.cos(4*pi*now_t/self.T)
                    X[y_n*i+j][k] = X[y_n*i+j][k]*(now_t<time)
                prev_time = time
        
        if test:
            return np.concatenate((X,z,w),axis=1), loc, partition_y, S_true_list, t
        else:
            return np.concatenate((X,z,w),axis=1), loc, y, ind, partition_y
        



class SimulatedFunctional:

    def __init__(self,setup):

        self.setup = setup
        self.x_grid_n = 100

    def generate_test(self,N,seed=0):

        np.random.seed(seed)
        sp1 = BrownianMotion(t=1, rng=np.random.default_rng(seed))
        sp2 = PoissonProcess(rate=10, rng=np.random.default_rng(seed))

        
        x_list = []
        y_list = []
        self.s = np.linspace(0,1,self.x_grid_n+1)[1:]

        for _ in range(N):
            x1 = sp1.sample(len(self.s)-1)
            x2 = sp2.sample(len(self.s)-1)
            x3 = np.random.normal()
            x4 = np.random.normal(x3,1)
            x = [[x1[i],x2[i],x3] for i in range(len(x1))]  
            if self.setup == 1:
                e = np.random.normal(0,1)
                y = np.sum((np.sin(x1)*x2+x2+3*x3)*(-np.log(self.s))*0.01)+e
            if self.setup == 2:
                e = np.random.normal(0,np.abs(x3))
                y = np.sum((np.sin(x1)*x2+x2+3*x3+3*x4)*(-np.log(self.s))*0.01)+e
            #if self.setup == 3:
                #mu = np.exp(np.sum((np.sin(x1)*x2+x2+x3)*(-np.log(self.s))*0.01)/10)
                #y = np.random.exponential(1/mu)
            #if self.setup == 4: 
                #y = np.sum((np.sin(x1)*x2+x2+x3+x4)*(-np.log(self.s))*0.01) + e

            x_list.append(x)
            y_list.append(y)

        return x_list, y_list

    def generate_data(self,N,seed,n_partition_y):
        np.random.seed(seed)
        sp1 = BrownianMotion(t=1, rng=np.random.default_rng(seed))
        sp2 = PoissonProcess(rate=10, rng=np.random.default_rng(seed))
        x_list = []
        y_list = []
        self.s = np.linspace(0,1,self.x_grid_n+1)[1:]

        for _ in range(N):
            x1 = sp1.sample(len(self.s)-1)
            x2 = sp2.sample(len(self.s)-1)
            x3 = np.random.normal()
            x4 = np.random.normal(x3,1)
            x = [[x1[i],x2[i],x3] for i in range(len(x1))] 
            if self.setup == 1:
                e = np.random.normal(0,1)
                y = np.sum((np.sin(x1)*x2+x2+3*x3)*(-np.log(self.s))*0.01)+e
            if self.setup == 2:
                e = np.random.normal(0,np.abs(x3))
                y = np.sum((np.sin(x1)*x2+x2+3*x3+3*x4)*(-np.log(self.s))*0.01)+e
            #if self.setup == 3:
                #mu = np.exp(np.sum((np.sin(x1)*x2+x2+x3)*(-np.log(self.s))*0.01)/10)
                #y = np.random.exponential(1/mu)
            #if self.setup == 4: # one time-varying covariate
                #y = np.sum((np.sin(x1)*x2+x2+x3+x4)*(-np.log(self.s))*0.01)+e

            x_list.append(x)
            y_list.append(y)

        #partition_y = np.sort(y_list)
        sorted_y = np.sort(y_list)
        partition_y = np.arange(sorted_y[0],sorted_y[-1],(sorted_y[-1]-sorted_y[0])/n_partition_y)

        y_n = len(partition_y)
        X = np.ones((N*y_n,len(self.s),len(x[0])))
        y = np.ones((N*y_n,1))
        loc = np.ones((N*y_n,1))
        ind = np.ones((N*y_n,1))


        for i in range(N):
            Y = y_list[i]
            x = x_list[i]
            prev_time = partition_y[0]-1
            for j, time in enumerate(partition_y):
                loc[y_n*i+j] = [time]
                X[y_n*i+j] = x
                if time < Y:
                    y[y_n*i+j] = [0]
                elif time >= Y:
                    y[y_n*i+j] = [1]
                #if time <= Y:
                if prev_time <= Y:
                    ind[y_n*i+j] = [1]
                else:
                    ind[y_n*i+j] = [0]
                prev_time = time
        
        return X, loc, y, ind, partition_y, x_list, y_list

    def expand_test(self,x_list,partition_y):
        y_n = len(partition_y)
        N = len(x_list)
        X = np.ones((N*y_n,len(x_list[0]),len(x_list[0][0])))
        loc = np.ones((N*y_n,1))

        for i in range(N):
            x = x_list[i]
            for j, time in enumerate(partition_y):
                loc[y_n*i+j] = [time]
                X[y_n*i+j] = x
        
        return X, loc

    def empirical_cdf(self,x_test,n_check=10,step=0.01):


        min_y, max_y = -1, 10

        t = np.arange(min_y,max_y,step)
        all_cdf = []

        for x in x_test[:n_check]:

            x1,x2,x3 = [xt[0] for xt in x], [xt[1] for xt in x], [xt[2] for xt in x]

            n = 10000
            cdf_rep = []
            for _ in range(5):
                cdf_list = []
                if self.setup == 1:
                    signal = np.sum((np.sin(x1)*x2+x2+3*x3[0])*(-np.log(self.s))*0.01)
                    e = np.random.normal(size=n)
                    y = signal + e
                elif self.setup == 2:
                    x4 = np.random.normal(x3[0],1,size=n)
                    signal = [np.sum((np.sin(x1)*x2+x2+3*x3[0]+3*x4[k])*(-np.log(self.s))*0.01) for k in range(n)]
                    e = np.random.normal(0,np.abs(x3[0]),size=n)
                    y = signal + e
                #elif self.setup == 3:
                    #y = np.random.exponential(1/np.exp(signal/10),size=n)
                #elif self.setup == 4:
                    #x4 = np.random.normal(1+0.5*x3[0],3/4,size=n)
                    #signal = [np.sum((np.sin(x1)*x2+x2+x3[0]+x4[k])*(-np.log(self.s))*0.01) for k in range(n)]
                    #e = np.random.normal(0,1,size=n)
                    #y = signal + e
                for y_grid in t:
                    cdf = sum(y<y_grid)/n
                    cdf_list.append(cdf)
                
                cdf_rep.append(cdf_list)
            all_cdf.append(np.mean(cdf_rep,axis=0))

        return np.array(all_cdf), t


class SimulatedDiscrete:

    def __init__(self):

        self.x_grid_n = 100

    def generate_test(self,N,seed=0):

        np.random.seed(seed)
        sp = BrownianMotion(t=1, rng=np.random.default_rng(seed))  
        x_list = []
        y_list = []
        self.s = np.linspace(0,1,self.x_grid_n+1)[1:]

        for _ in range(N):
            x1 = sp.sample(len(self.s)-1)
            w = np.random.normal()
            
            w = [w for _ in range(len(x1))]
            x = [[x1[i],w[i]] for i in range(len(x1))] 
            lam = np.sum((np.sin(x1)+w)**2*(-np.log(self.s))*0.02)
            y = np.random.poisson(lam)

            x_list.append(x)
            y_list.append(y)

        return x_list, y_list

    def generate_data(self,N,seed):
        np.random.seed(seed)
        sp = BrownianMotion(t=1, rng=np.random.default_rng(seed))
        x_list = []
        y_list = []
        self.s = np.linspace(0,1,self.x_grid_n+1)[1:]

        for _ in range(N):
            x1 = sp.sample(len(self.s)-1)
            w = np.random.normal()
            w = [w for _ in range(len(x1))]
            x = [[x1[i],w[i]] for i in range(len(x1))] 
            lam = np.sum((np.sin(x1)+w)**2*(-np.log(self.s))*0.02)
            y = np.random.poisson(lam)

            x_list.append(x)
            y_list.append(y)

        partition_y = list(range(max(y_list)+1))

        y_n = len(partition_y)
        X = np.ones((N*y_n,len(self.s),len(x[0])))
        y = np.ones((N*y_n,1))
        loc = np.ones((N*y_n,1))
        ind = np.ones((N*y_n,1))


        for i in range(N):
            Y = y_list[i]
            x = x_list[i]
            for j, time in enumerate(partition_y):
                loc[y_n*i+j] = [time]
                X[y_n*i+j] = x
                if time < Y:
                    y[y_n*i+j] = [0]
                elif time >= Y:
                    y[y_n*i+j] = [1]
                if time <= Y:
                    ind[y_n*i+j] = [1]
                else:
                    ind[y_n*i+j] = [0]
        
        return X, loc, y, ind, partition_y

    def expand_test(self,x_list,partition_y):
        y_n = len(partition_y)
        N = len(x_list)
        X = np.ones((N*y_n,len(x_list[0]),len(x_list[0][0])))
        loc = np.ones((N*y_n,1))

        for i in range(N):
            x = x_list[i]
            for j, time in enumerate(partition_y):
                loc[y_n*i+j] = [time]
                X[y_n*i+j] = x
        
        return X, loc

    def empirical_cdf(self,x_test,n_check=10,step=1):


        min_y, max_y = 0, 13

        t = np.arange(min_y,max_y,step)
        all_cdf = []

        for x in x_test[:n_check]:
        
            x, w = [xt[0] for xt in x], [xt[1] for xt in x]
            lam = np.sum((np.sin(x)+w)**2*(-np.log(self.s))*0.02)

            n = 10000
            cdf_rep = []
            for _ in range(10):
                cdf_list = []
                y = np.random.poisson(lam,size=n)
                for y_grid in t:
                    cdf = sum(y<=y_grid)/n
                    cdf_list.append(cdf)
                
                cdf_rep.append(cdf_list)
            all_cdf.append(np.mean(cdf_rep,axis=0))

        return np.array(all_cdf), t
    

def expand_data(temp,w,y,n_partition_y=500,partition_y=None):
    N = len(y)
    m = len(temp[0])
    sorted_y = np.sort(np.unique(y))
    if partition_y is None:
        #partition_y = np.arange(sorted_y[0],sorted_y[-1],(sorted_y[-1]-sorted_y[0])/n_partition_y)
        partition_y = sorted_y
    y_n = len(partition_y)
    X = np.ones((N*y_n,m,2))
    fail = np.ones((N*y_n,1))
    loc = np.ones((N*y_n,1))
    ind = np.ones((N*y_n,1))


    for i in range(N):
        Y = y[i]
        tempi = temp[i]
        wi = w[i]
        x = [[tempi[j],wi] for j in range(len(tempi))] 
        prev_time = partition_y[0]-1
        for j, time in enumerate(partition_y):
            loc[y_n*i+j] = [time]
            X[y_n*i+j] = x
            if time < Y:
                fail[y_n*i+j] = [0]
            elif time >= Y:
                fail[y_n*i+j] = [1]
            if time <= Y:
            #if prev_time <= Y:
                ind[y_n*i+j] = [1]
            else:
                ind[y_n*i+j] = [0]
            prev_time = time

    return X, loc, fail, ind, partition_y