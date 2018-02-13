#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:32:19 2018

@author: Paris
"""

import autograd.numpy as np
from autograd import grad
from optimizers import Adam
import timeit

class NeuralNetwork:
    # Initialize the class
    def __init__(self, X, Y, layers):
        
        # Normalize data
        self.Xmean, self.Xstd = X.mean(0), X.std(0)
        self.Ymean, self.Ystd = Y.mean(0), Y.std(0)
        X = (X - self.Xmean) / self.Xstd
        Y = (Y - self.Ymean) / self.Ystd
            
        self.X = X
        self.Y = Y
        self.layers = layers
        
        # Define and initialize neural network
        self.params = self.initialize_NN(self.layers)
        
        # Total number of parameters
        self.num_params = self.params.shape[0]
        
        # Define optimizer
        self.optimizer = Adam(self.num_params, lr = 1e-3)
        
        # Define gradient function using autograd 
        self.grad_loss = grad(self.loss)
        
        
    # Initializes the network weights and biases using Xavier initialization
    def initialize_NN(self, Q):
        params = np.array([])
        num_layers = len(Q)
        for layer in range(0,num_layers-1):
            weights = -np.sqrt(6.0/(Q[layer]+Q[layer+1])) + 2.0*np.sqrt(6.0/(Q[layer]+Q[layer+1]))*np.random.rand(Q[layer],Q[layer+1])
            biases = np.zeros((1,Q[layer+1]))
            params = np.concatenate([params, weights.ravel(), biases.ravel()])        
        return params
        
    
    # Evaluates the forward pass
    def forward_pass(self, X, Q, params):
        H = X
        idx_3 = 0
        num_layers = len(self.layers) 
        # All layers up to last
        for layer in range(0,num_layers-2):        
            idx_1 = idx_3
            idx_2 = idx_1 + Q[layer]*Q[layer+1]
            idx_3 = idx_2 + Q[layer+1]
            weights = np.reshape(params[idx_1:idx_2], (Q[layer],Q[layer+1]))
            biases = np.reshape(params[idx_2:idx_3], (1,Q[layer+1]))
            H = np.tanh(np.matmul(H,weights) + biases)
            
        # Last linear layer
        idx_1 = idx_3
        idx_2 = idx_1 + Q[-2]*Q[-1]
        idx_3 = idx_2 + Q[-1]
        weights = np.reshape(params[idx_1:idx_2], (Q[-2],Q[-1]))
        biases = np.reshape(params[idx_2:idx_3], (1,Q[-1]))
        mu = np.matmul(H,weights) + biases
                
        return mu
    
    
    # Evaluates the mean square error loss
    def loss(self, params):
        X = self.X_batch
        Y = self.Y_batch                          
        mu = self.forward_pass(X, self.layers, params)                
        return np.mean((Y-mu)**2)
    
    
    # Fetches a mini-batch of data
    def fetch_minibatch(self,X, Y, N_batch):
        N = X.shape[0]
        idx = np.random.choice(N, N_batch, replace=False)
        X_batch = X[idx,:]
        Y_batch = Y[idx,:]        
        return X_batch, Y_batch
    
    
    # Trains the model by minimizing the MSE loss
    def train(self, nIter = 10000, batch_size = 100):   

        start_time = timeit.default_timer()            
        for it in range(nIter):
            # Fetch minibatch
            self.X_batch, self.Y_batch = self.fetch_minibatch(self.X, self.Y, batch_size)
            
            # Evaluate loss using current parameters
            params = self.params
            loss = self.loss(params)
          
            # Update parameters
            grad_params = self.grad_loss(params)
            self.params = self.optimizer.step(params, grad_params)
            
            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss, elapsed))
                start_time = timeit.default_timer()
            
      
    # Evaluates predictions at test points              
    def predict(self, X_star): 
        # Normalize inputs
        X_star = (X_star - self.Xmean) / self.Xstd            
        y_star = self.forward_pass(X_star, self.layers, self.params)
        # De-normalize outputs
        y_star = y_star*self.Ystd + self.Ymean            
        return y_star
    

class LinearRegression:
    """
        Linear regression model: y = (w.T)*x + \epsilon
        p(y|x,theta) ~ N(y|(w.T)*x, sigma^2), theta = (w, sigma^2)
    """
    # Initialize model class
    def __init__(self, X, Y):
        
        # Normalize data
        self.Xmean, self.Xstd = X.mean(0), X.std(0)
        self.Ymean, self.Ystd = Y.mean(0), Y.std(0)
        X = (X - self.Xmean) / self.Xstd
        Y = (Y - self.Ymean) / self.Ystd
      
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
      
        # Randomly initialize weights and noise variance
        w = np.random.randn(X.shape[1], Y.shape[1])
        sigma_sq = np.array([np.log([1e-3])])
      
        # Concatenate all parameters in a single vector
        self.theta = np.concatenate([w.flatten(), sigma_sq.flatten()])
      
        # Count total number of parameters
        self.num_params = self.theta.shape[0]
      
        # Define optimizer
        self.optimizer = Adam(self.num_params, lr = 1e-3)
      
        # Define loss gradient function using autograd
        self.grad_loss = grad(self.loss)
      
    
    # Evaluates the forward prediction of the model
    def forward_pass(self, X,w):
        y = np.matmul(X, w)
        return y


    # Evaluates the negative log-likelihood loss, i.e. -log p(y|x,theta)
    def loss(self, theta): 
        X = self.X_batch
        Y = self.Y_batch
        # Fetch individual parameters from the theta vector and reshape if needed
        w = np.reshape(theta[:-1],(self.X.shape[1], self.Y.shape[1]))
        sigma_sq = np.exp(theta[-1])
        # Evaluate the model's prediction
        Y_pred = self.forward_pass(X, w)
        # Compute the loss
        NLML = 0.5 * self.n * np.log(2.0*np.pi*sigma_sq) + \
                 0.5 * (np.sum(Y - Y_pred)**2) / sigma_sq
        return NLML
        
    
    # Fetches a mini-batch of data
    def fetch_minibatch(self,X, Y, N_batch):
        N = X.shape[0]
        idx = np.random.choice(N, N_batch, replace=False)
        X_batch = X[idx,:]
        Y_batch = Y[idx,:]        
        return X_batch, Y_batch
    
    
    # Trains the model by minimizing the MSE loss
    def train(self, nIter = 10000, batch_size = 100):   

        start_time = timeit.default_timer()            
        for it in range(nIter):
            # Fetch minibatch
            self.X_batch, self.Y_batch = self.fetch_minibatch(self.X, self.Y, batch_size)
            
            # Evaluate loss using current parameters
            theta = self.theta
            loss = self.loss(theta)
          
            # Update parameters
            grad_theta = self.grad_loss(theta)
            self.theta = self.optimizer.step(theta, grad_theta)
            
            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss, elapsed))
                start_time = timeit.default_timer()
      
        
    # Evaluates predictions at test points          
    def predict(self, X_star):
        # Normalize inputs
        X_star = (X_star - self.Xmean) / self.Xstd 
        w = np.reshape(self.theta[:-1],(self.X.shape[1], self.Y.shape[1]))
        y_star = self.forward_pass(X_star, w)
        # De-normalize outputs
        y_star = y_star*self.Ystd + self.Ymean 
        return y_star
    

