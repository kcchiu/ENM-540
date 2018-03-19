#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 19:39:51 2018

@author: Paris
"""

import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize

# A minimal Gaussian process class
class BeamGP:
    # Initialize the class
    def __init__(self, X_u, y_u, X_f, y_f, 
                 k_uu, k_uu1, k_uu2, k_uu3, k_uf,
                 k_u1u1, k_u1u2, k_u1u3, k_u1f,
                 k_u2u2, k_u2u3, k_u2f,
                 k_u3u3, k_u3f,
                 k_ff): 
        
        self.D = X_f.shape[1]
        
        self.X_u = X_u
        self.y_u = y_u
        
        self.X_f = X_f
        self.y_f = y_f
        
        self.hyp = self.init_params()        
        
        self.k_uu = k_uu
        self.k_uu1 = k_uu1
        self.k_uu2 = k_uu2
        self.k_uu3 = k_uu3
        self.k_uf = k_uf
        
        self.k_u1u1 = k_u1u1
        self.k_u1u2 = k_u1u2
        self.k_u1u3 = k_u1u3
        self.k_u1f = k_u1f
        
        self.k_u2u2 = k_u2u2
        self.k_u2u3 = k_u2u3
        self.k_u2f = k_u2f
        
        self.k_u3u3 = k_u3u3
        self.k_u3f = k_u3f
        
        self.k_ff = k_ff
                       
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
    
    
    # Computes the negative log-marginal likelihood
    def likelihood(self, hyp):
        X_u = self.X_u
        y_u = self.y_u
        
        X_f = self.X_f
        y_f = self.y_f
        
        y = np.vstack((y_u, y_f))

        N = y.shape[0]
        N_f = y_f.shape[0]
        
        theta = hyp[:-1]
        sigma_n = np.exp(hyp[-1])
               
        K_uu = self.k_uu(X_u[0:1,:], X_u[0:1,:], theta)
        K_uu1 = self.k_uu1(X_u[0:1,:], X_u[1:2,:], theta)
        K_uu2 = self.k_uu2(X_u[0:1,:], X_u[2:3,:], theta)
        K_uu3 = self.k_uu3(X_u[0:1,:], X_u[3:4,:], theta)
        K_uf = self.k_uf(X_u[0:1,:], X_f, theta)
        
        K_u1u1 = self.k_u1u1(X_u[1:2,:], X_u[1:2,:], theta)
        K_u1u2 = self.k_u1u2(X_u[1:2,:], X_u[2:3,:], theta)
        K_u1u3 = self.k_u1u3(X_u[1:2,:], X_u[3:4,:], theta)
        K_u1f = self.k_u1f(X_u[1:2,:], X_f, theta)
        
        K_u2u2 = self.k_u2u2(X_u[2:3,:], X_u[2:3,:], theta)
        K_u2u3 = self.k_u2u3(X_u[2:3,:], X_u[3:4,:], theta)
        K_u2f = self.k_u2f(X_u[2:3,:], X_f, theta)
        
        K_u3u3 = self.k_u3u3(X_u[3:4,:], X_u[3:4,:], theta)
        K_u3f = self.k_u3f(X_u[3:4,:], X_f, theta)
        
        K_ff = self.k_ff(X_f, X_f, theta) + np.eye(N_f)*sigma_n
        
        K = np.vstack((np.hstack((K_uu, K_uu1, K_uu2, K_uu3, K_uf)),
                       np.hstack((K_uu1.T, K_u1u1, K_u1u2, K_u1u3, K_u1f)),
                       np.hstack((K_uu2.T, K_u1u2.T, K_u2u2, K_u2u3, K_u2f)),
                       np.hstack((K_uu3.T, K_u1u3.T, K_u2u3.T, K_u3u3, K_u3f)),
                       np.hstack((K_uf.T, K_u1f.T, K_u2f.T, K_u3f.T, K_ff))))
        
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
    def predict_u(self,X_star):
        X_u = self.X_u
        y_u = self.y_u
        
        X_f = self.X_f
        y_f = self.y_f
        
        y = np.vstack((y_u, y_f))

        L = self.L
                
        theta = self.hyp[:-1]
        
        K_uu = self.k_uu(X_star, X_u[0:1,:], theta)
        K_uu1 = self.k_uu1(X_star, X_u[1:2,:], theta)
        K_uu2 = self.k_uu2(X_star, X_u[2:3,:], theta)
        K_uu3 = self.k_uu3(X_star, X_u[3:4,:], theta)
        K_uf = self.k_uf(X_star, X_f, theta)
        psi = np.hstack((K_uu, K_uu1, K_uu2, K_uu3, K_uf))
        
        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L,y))
        pred_u_star = np.matmul(psi,alpha)

        beta = np.linalg.solve(np.transpose(L), np.linalg.solve(L,psi.T))
        var_u_star = self.k_uu(X_star, X_star, theta) - np.matmul(psi,beta)
        
        return pred_u_star, var_u_star
    
    
    # Return posterior mean and variance at a set of test points
    def predict_f(self,X_star):
        X_u = self.X_u
        y_u = self.y_u
        
        X_f = self.X_f
        y_f = self.y_f
        
        y = np.vstack((y_u, y_f))

        L = self.L
                
        theta = self.hyp[:-1]
        
        K_uf = self.k_uf(X_u[0:1,:], X_star, theta)
        K_u1f = self.k_u1f(X_u[1:2,:], X_star, theta)
        K_u2f = self.k_u2f(X_u[2:3,:], X_star, theta)
        K_u3f = self.k_u3f(X_u[3:4,:], X_star, theta)
        K_ff = self.k_ff(X_star, X_f, theta)
        psi = np.hstack((K_uf.T, K_u1f.T, K_u2f.T, K_u3f.T, K_ff))
        
        alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L,y))
        pred_u_star = np.matmul(psi,alpha)

        beta = np.linalg.solve(np.transpose(L), np.linalg.solve(L,psi.T))
        var_u_star = self.k_ff(X_star, X_star, theta) - np.matmul(psi,beta)
        
        return pred_u_star, var_u_star
    
    
    def DE_UCB_max(self, X_star):      
        u, v_u = self.predict_u(X_star)
        f, v_f = self.predict_f(X_star)      
        v_u = np.abs(np.diag(v_u))[:,None]
        v_f = np.abs(np.diag(v_f))[:,None]
        nn = np.linalg.norm(np.sqrt(v_u))/np.linalg.norm(np.sqrt(v_f))
        acq = u + (np.sqrt(v_f)*nn + np.sqrt(v_u))
        return acq
    
    
    def DE_UCB_min(self, X_star):      
        u, v_u = self.predict_u(X_star)
        f, v_f = self.predict_f(X_star)      
        v_u = np.abs(np.diag(v_u))[:,None]
        v_f = np.abs(np.diag(v_f))[:,None]
        nn = np.linalg.norm(np.sqrt(v_u))/np.linalg.norm(np.sqrt(v_f))
        acq = u - (np.sqrt(v_f)*nn + np.sqrt(v_u))
        return acq
 

