#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 20:56:05 2017

@author: Paris
"""

import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from scipy.stats import norm

# A minimal Gaussian process class
class GPRegression:
    # Initialize the class
    def __init__(self, X, y): 
        
        # Normalize data
        self.Xmean, self.Xstd = X.mean(0), X.std(0)
        self.Ymean, self.Ystd = y.mean(0), y.std(0)
        X = (X - self.Xmean) / self.Xstd
        y = (y - self.Ymean) / self.Ystd
              
        self.D = X.shape[1]
        self.X = X
        self.y = y
        
        self.hyp = self.init_params()
               
        self.jitter = 1e-8
        
        self.likelihood(self.hyp)
        print("Total number of parameters: %d" % (self.hyp.shape[0]))

    # Initialize hyper-parameters        
    def init_params(self):
        # Kernel hyper-parameters
        hyp = np.log(np.ones(self.D+1))
        # Noise variance
        logsigma_n = np.array([-4.0])
        hyp = np.concatenate([hyp, logsigma_n])
        return hyp
    
    
    # A simple vectorized rbf kernel
    def kernel(self,x,xp,hyp):
        output_scale = np.exp(hyp[0])
        lengthscales = np.exp(hyp[1:])
        diffs = np.expand_dims(x /lengthscales, 1) - \
                np.expand_dims(xp/lengthscales, 0)
        return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2))
        
    
    # Computes the negative log-marginal likelihood
    def likelihood(self, hyp):
        X = self.X
        y = self.y

        N = y.shape[0]
        
        theta = hyp[:-1]
        sigma_n = np.exp(hyp[-1])
               
        K = self.kernel(X, X, theta) + np.eye(N)*sigma_n
        L = np.linalg.cholesky(K + np.eye(N)*self.jitter) 
        self.L = L
        
        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L,y))    
        NLML = 0.5*np.matmul(np.transpose(y),alpha) + \
               np.sum(np.log(np.diag(L))) + 0.5*np.log(2.*np.pi)*N  
        return NLML[0,0]
    
            
    #  Prints the negative log-marginal likelihood at each training step         
    def callback(self,params):
        print("Log likelihood {}".format(self.likelihood(params)))
        

    # Minimizes the negative log-marginal likelihood using L-BFGS
    def train(self):
        result = minimize(value_and_grad(self.likelihood), self.hyp, jac=True, 
                          method='L-BFGS-B', callback=self.callback)
        self.hyp = result.x
        
        
    # Return posterior mean and variance at a set of test points
    def predict(self,X_star):
        # Normalize data
        X_star = (X_star - self.Xmean) / self.Xstd
               
        X = self.X
        y = self.y
       
        L = self.L
                
        theta = self.hyp[:-1]
        
        psi = self.kernel(X_star, X, theta)

        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L,y))
        pred_u_star = np.matmul(psi,alpha)

        beta = np.linalg.solve(np.transpose(L), np.linalg.solve(L,psi.T))
        var_u_star = self.kernel(X_star, X_star, theta) - np.matmul(psi,beta)
        
        # De-normalize
        pred_u_star = pred_u_star*self.Ystd + self.Ymean
        var_u_star = var_u_star*self.Ystd**2
        
        return pred_u_star, var_u_star
    
    
    def draw_prior_samples(self, X_star, N_samples = 1):
        # Normalize data
        X_star = (X_star - self.Xmean) / self.Xstd            
        N = X_star.shape[0]        
        theta = self.hyp[:-1]    
        K = self.kernel(X_star, X_star, theta)     
        samples = np.random.multivariate_normal(np.zeros(N), K, N_samples).T
        # De-normalize
        samples = samples*self.Ystd + self.Ymean
        return samples
                   
                          
    def draw_posterior_samples(self, X_star, N_samples = 1):
        # Normalize data
        X_star = (X_star - self.Xmean) / self.Xstd
        
        X = self.X
        y = self.y
       
        L = self.L
                
        theta = self.hyp[:-1]
        
        psi = self.kernel(X_star, X, theta)

        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L,y))
        pred_u_star = np.matmul(psi,alpha)

        beta = np.linalg.solve(np.transpose(L), np.linalg.solve(L,psi.T))
        var_u_star = self.kernel(X_star, X_star, theta) - np.matmul(psi,beta)
        
        samples = np.random.multivariate_normal(pred_u_star.flatten(), 
                                             var_u_star, N_samples).T
                                                
        # De-normalize
        samples = samples*self.Ystd + self.Ymean
        
        return samples                                      
         

    def ExpectedImprovement(self, X_star):
        # Normalize data
        X_star = (X_star - self.Xmean) / self.Xstd
              
        X = self.X
        y = self.y
       
        L = self.L
                
        theta = self.hyp[:-1]
        
        psi = self.kernel(X_star, X, theta)

        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L,y))
        pred_u_star = np.matmul(psi,alpha)

        beta = np.linalg.solve(np.transpose(L), np.linalg.solve(L,psi.T))
        var_u_star = self.kernel(X_star, X_star, theta) - np.matmul(psi,beta)
        var_u_star = np.abs(np.diag(var_u_star))[:,None]
        
        # Expected Improvement
        best = np.min(y)
        Z = (best - pred_u_star)/var_u_star
        EI_acq = (best - pred_u_star)*norm.cdf(Z) + var_u_star*norm.pdf(Z)

        return EI_acq
    