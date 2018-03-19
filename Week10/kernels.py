#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 18:55:31 2018

@author: Paris
"""

import sympy as sp
import autograd.numpy as np
from autograd.scipy.special import erf
from sympy.printing.lambdarepr import NumPyPrinter


def Beam_kernels():
    
    def operator(k, x):
        return sp.diff(k,x,4)
    
    ############################## Symbolic ##############################
    
    autograd_modules = [{'And': np.logical_and, 'Or': np.logical_or, 
                         'erf': erf}, np]

    # Define kernels
    x, x_prime = sp.symbols('x x_prime')
    sigma, theta = sp.symbols('sigma theta')
    
    k_uu = sigma*sp.exp(-0.5*((x-x_prime)**2/theta))
    k_uu1 = sp.diff(k_uu, x_prime, 1)
    k_uu2 = sp.diff(k_uu, x_prime, 2)
    k_uu3 = sp.diff(k_uu, x_prime, 3)
    k_uf = operator(k_uu, x_prime)
    
    k_u1u1 = sp.diff(sp.diff(k_uu, x_prime, 1), x, 1)
    k_u1u2 = sp.diff(sp.diff(k_uu, x_prime, 2), x, 1)
    k_u1u3 = sp.diff(sp.diff(k_uu, x_prime, 3), x, 1)
    k_u1f =  sp.diff(operator(k_uu, x_prime),  x, 1)
       
    k_u2u2 = sp.diff(sp.diff(k_uu, x_prime, 2), x, 2)
    k_u2u3 = sp.diff(sp.diff(k_uu, x_prime, 3), x, 2)
    k_u2f =  sp.diff(operator(k_uu, x_prime),  x, 2)
        
    k_u3u3 = sp.diff(sp.diff(k_uu, x_prime, 3), x, 3)
    k_u3f =  sp.diff(operator(k_uu, x_prime),  x, 3)
    
    k_ff = operator(operator(k_uu, x_prime), x)
    
    ############################## Lambdify ##############################
    
    ########################## Row 1 ##################################
    
    lambda_k_uu = sp.lambdify((x, x_prime, sigma, theta), k_uu, 
                              dummify=True, modules=autograd_modules, 
                              printer=NumPyPrinter)
    
    lambda_k_uu1 = sp.lambdify((x, x_prime, sigma, theta), k_uu1, 
                               dummify=True, modules=autograd_modules, 
                               printer=NumPyPrinter)
    
    lambda_k_uu2 = sp.lambdify((x, x_prime, sigma, theta), k_uu2, 
                               dummify=True, modules=autograd_modules, 
                               printer=NumPyPrinter)
    
    lambda_k_uu3 = sp.lambdify((x, x_prime, sigma, theta), k_uu3, 
                               dummify=True, modules=autograd_modules, 
                               printer=NumPyPrinter)
       
    
    lambda_k_uf = sp.lambdify((x, x_prime, sigma, theta), k_uf, 
                              dummify=True, modules=autograd_modules, 
                              printer=NumPyPrinter)
    
    ########################## Row 2 ##################################
    
    lambda_k_u1u1 = sp.lambdify((x, x_prime, sigma, theta), k_u1u1, 
                                dummify=True, modules=autograd_modules, 
                                printer=NumPyPrinter)
    
    lambda_k_u1u2 = sp.lambdify((x, x_prime, sigma, theta), k_u1u2, 
                                dummify=True, modules=autograd_modules, 
                                printer=NumPyPrinter)
    
    lambda_k_u1u3 = sp.lambdify((x, x_prime, sigma, theta), k_u1u3, 
                                dummify=True, modules=autograd_modules, 
                                printer=NumPyPrinter)
    
    lambda_k_u1f =  sp.lambdify((x, x_prime, sigma, theta),  k_u1f, 
                                dummify=True, modules=autograd_modules, 
                                printer=NumPyPrinter)
    
     ########################## Row 3 ##################################
       
    lambda_k_u2u2 = sp.lambdify((x, x_prime, sigma, theta), k_u2u2, 
                                dummify=True, modules=autograd_modules, 
                                printer=NumPyPrinter)
    
    lambda_k_u2u3 = sp.lambdify((x, x_prime, sigma, theta), k_u2u3, 
                                dummify=True, modules=autograd_modules, 
                                printer=NumPyPrinter)
    
    lambda_k_u2f =  sp.lambdify((x, x_prime, sigma, theta),  k_u2f, 
                                dummify=True, modules=autograd_modules, 
                                printer=NumPyPrinter)
     
    ########################## Row 4 ##################################
       
    lambda_k_u3u3 = sp.lambdify((x, x_prime, sigma, theta), k_u3u3, 
                                dummify=True, modules=autograd_modules, 
                                printer=NumPyPrinter)
    
    lambda_k_u3f =  sp.lambdify((x, x_prime, sigma, theta),  k_u3f, 
                                dummify=True, modules=autograd_modules, 
                                printer=NumPyPrinter)

    ########################## Row 5 ##################################
   
    lambda_k_ff = sp.lambdify((x, x_prime, sigma, theta), k_ff, 
                              dummify=True, modules=autograd_modules, 
                              printer=NumPyPrinter)
    
    
    ############################## Vectorization ##############################
    
    ########################## Row 1 ##################################

    def k_uu(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
                    
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_uu(x, x_prime, sigma, theta)
    
    def k_uu1(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
                    
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_uu1(x, x_prime, sigma, theta)
    
    def k_uu2(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
                    
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_uu2(x, x_prime, sigma, theta)
    
    def k_uu3(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
                    
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_uu3(x, x_prime, sigma, theta)
    
    def k_uf(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
                    
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_uf(x, x_prime, sigma, theta)
    
    
    ########################## Row 2 ##################################

    def k_u1u1(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
                    
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_u1u1(x, x_prime, sigma, theta)
    
    def k_u1u2(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
                    
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_u1u2(x, x_prime, sigma, theta)
    
    def k_u1u3(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
                    
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_u1u3(x, x_prime, sigma, theta)
    
    def k_u1f(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
                    
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_u1f(x, x_prime, sigma, theta)
    
    ########################## Row 3 ##################################

    def k_u2u2(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
                    
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_u2u2(x, x_prime, sigma, theta)

    def k_u2u3(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
                    
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_u2u3(x, x_prime, sigma, theta) 
    
    def k_u2f(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
                    
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_u2f(x, x_prime, sigma, theta) 
    
    ########################## Row 4 ##################################

    def k_u3u3(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
                    
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_u3u3(x, x_prime, sigma, theta)   
    
    def k_u3f(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
                    
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_u3f(x, x_prime, sigma, theta)   
    
    ########################## Row 5 ##################################

    def k_ff(x, x_prime, hyp):
        sigma = np.exp(hyp[0])
        theta = np.exp(hyp[1])
                    
        N = x.shape[0]
        N_prime = x_prime.shape[0]
    
        x = np.matmul(x, np.ones((1,N_prime)))
        x_prime = np.matmul(np.ones((N,1)),x_prime.T)

        return lambda_k_ff(x, x_prime, sigma, theta)   

    
    ########################## Return ##################################

    return k_uu, k_uu1, k_uu2, k_uu3, k_uf, \
           k_u1u1, k_u1u2, k_u1u3, k_u1f, \
           k_u2u2, k_u2u3, k_u2f, \
           k_u3u3, k_u3f, \
           k_ff
           
           
           
    


                    
    


