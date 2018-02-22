#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:26:05 2018

@author: Paris
"""

import autograd.numpy as np
from autograd import grad
from optimizers import Adam
import timeit

class RNN:
    # Initialize the class
    def __init__(self, X, Y, hidden_dim):
               
        # X has the form lags x data x dim
        # Y has the form data x dim
        
        self.X = X
        self.Y = Y
        self.X_dim = X.shape[-1]
        self.Y_dim = Y.shape[-1]
        self.hidden_dim = hidden_dim
        self.lags = X.shape[0]
        
        # Define and initialize neural network
        self.params = self.initialize_RNN()
        
        # Total number of parameters
        self.num_params = self.params.shape[0]
        
        # Define optimizer
        self.optimizer = Adam(self.num_params, lr = 1e-3)
        
        # Define gradient function using autograd 
        self.grad_loss = grad(self.loss)
        
        
    # Initializes the network weights and biases using Xavier initialization
    def initialize_RNN(self):
        hyp = np.array([])
        Q = self.hidden_dim
            
        U = -np.sqrt(6.0/(self.X_dim+Q)) + 2.0*np.sqrt(6.0/(self.X_dim+Q))*np.random.rand(self.X_dim,Q)
        b = np.zeros((1,Q))
        W = np.eye(Q)
        hyp = np.concatenate([hyp, U.ravel(), b.ravel(), W.ravel()])            
        
        V = -np.sqrt(6.0/(Q+self.Y_dim)) + 2.0*np.sqrt(6.0/(Q+self.Y_dim))*np.random.rand(Q,self.Y_dim)
        c = np.zeros((1,self.Y_dim))
        hyp = np.concatenate([hyp, V.ravel(), c.ravel()])
    
        return hyp
    
    
    def forward_pass(self, X, hyp):     
        Q = self.hidden_dim
        H = np.zeros((X.shape[1],Q))
        
        idx_1 = 0
        idx_2 = idx_1 + self.X_dim*Q
        idx_3 = idx_2 + Q
        idx_4 = idx_3 + Q*Q
        U = np.reshape(hyp[idx_1:idx_2], (self.X_dim,Q))
        b = np.reshape(hyp[idx_2:idx_3], (1,Q))
        W = np.reshape(hyp[idx_3:idx_4], (Q,Q))
        
        for i in range(0, self.lags):
            H = np.tanh(np.matmul(H,W) + np.matmul(X[i,:,:],U) + b)
                
        idx_1 = idx_4
        idx_2 = idx_1 + Q*self.Y_dim
        idx_3 = idx_2 + self.Y_dim
        V = np.reshape(hyp[idx_1:idx_2], (Q,self.Y_dim))
        c = np.reshape(hyp[idx_2:idx_3], (1,self.Y_dim))
        Y = np.matmul(H,V) + c
        
        return Y
    
    
    # Evaluates the mean square error loss
    def loss(self, params):
        X = self.X_batch
        Y = self.Y_batch                          
        mu = self.forward_pass(X, params)                
        return np.mean((Y-mu)**2)
    
    
    
    # Fetches a mini-batch of data
    def fetch_minibatch_rnn(self, X,Y,N_batch):
        N = X.shape[1]
        idx = np.random.choice(N, N_batch, replace=False)
        X_batch = X[:,idx,:]
        Y_batch = Y[idx,:]
        return X_batch, Y_batch
    
    
    # Trains the model by minimizing the MSE loss
    def train(self, nIter = 10000, batch_size = 100):   

        start_time = timeit.default_timer()            
        for it in range(nIter):
            # Fetch minibatch
            self.X_batch, self.Y_batch = self.fetch_minibatch_rnn(self.X, self.Y, batch_size)
            
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
        y_star = self.forward_pass(X_star, self.params)
        return y_star
    


class LSTM:
    # Initialize the class
    def __init__(self, X, Y, hidden_dim):
               
        # X has the form lags x data x dim
        # Y has the form data x dim
        
        self.X = X
        self.Y = Y
        self.X_dim = X.shape[-1]
        self.Y_dim = Y.shape[-1]
        self.hidden_dim = hidden_dim
        self.lags = X.shape[0]
        
        # Define and initialize neural network
        self.params = self.initialize_LSTM()
        
        # Total number of parameters
        self.num_params = self.params.shape[0]
        
        # Define optimizer
        self.optimizer = Adam(self.num_params, lr = 1e-3)
        
        # Define gradient function using autograd 
        self.grad_loss = grad(self.loss)
        
        
    # Initializes the network weights and biases using Xavier initialization
    def initialize_LSTM(self):
        hyp = np.array([])
        Q = self.hidden_dim
        
        # Forget Gate
        U_f = -np.sqrt(6.0/(self.X_dim+Q)) + 2.0*np.sqrt(6.0/(self.X_dim+Q))*np.random.rand(self.X_dim,Q)
        b_f = np.zeros((1,Q))
        W_f = np.eye(Q)
        hyp = np.concatenate([hyp, U_f.ravel(), b_f.ravel(), W_f.ravel()])
        
        # Input Gate
        U_i = -np.sqrt(6.0/(self.X_dim+Q)) + 2.0*np.sqrt(6.0/(self.X_dim+Q))*np.random.rand(self.X_dim,Q)
        b_i = np.zeros((1,Q))
        W_i = np.eye(Q)
        hyp = np.concatenate([hyp, U_i.ravel(), b_i.ravel(), W_i.ravel()])

        # Update Cell State
        U_s = -np.sqrt(6.0/(self.X_dim+Q)) + 2.0*np.sqrt(6.0/(self.X_dim+Q))*np.random.rand(self.X_dim,Q)
        b_s = np.zeros((1,Q))
        W_s = np.eye(Q)
        hyp = np.concatenate([hyp, U_s.ravel(), b_s.ravel(), W_s.ravel()])

        # Ouput Gate
        U_o = -np.sqrt(6.0/(self.X_dim+Q)) + 2.0*np.sqrt(6.0/(self.X_dim+Q))*np.random.rand(self.X_dim,Q)
        b_o = np.zeros((1,Q))
        W_o = np.eye(Q)
        hyp = np.concatenate([hyp, U_o.ravel(), b_o.ravel(), W_o.ravel()])

        V = -np.sqrt(6.0/(Q+self.Y_dim)) + 2.0*np.sqrt(6.0/(Q+self.Y_dim))*np.random.rand(Q,self.Y_dim)
        c = np.zeros((1,self.Y_dim))
        hyp = np.concatenate([hyp, V.ravel(), c.ravel()])
    
        return hyp
    
    
    def sigmoid(self, x):
        return 0.5*(np.tanh(x) + 1.0)
    
        
    def forward_pass(self, X, hyp):     
        Q = self.hidden_dim
        H = np.zeros((X.shape[1],Q))
        S = np.zeros((X.shape[1],Q))
        
        # Forget Gate
        idx_1 = 0
        idx_2 = idx_1 + self.X_dim*Q
        idx_3 = idx_2 + Q
        idx_4 = idx_3 + Q*Q
        U_f = np.reshape(hyp[idx_1:idx_2], (self.X_dim,Q))
        b_f = np.reshape(hyp[idx_2:idx_3], (1,Q))
        W_f = np.reshape(hyp[idx_3:idx_4], (Q,Q))
        
        # Input Gate
        idx_1 = idx_4
        idx_2 = idx_1 + self.X_dim*Q
        idx_3 = idx_2 + Q
        idx_4 = idx_3 + Q*Q
        U_i = np.reshape(hyp[idx_1:idx_2], (self.X_dim,Q))
        b_i = np.reshape(hyp[idx_2:idx_3], (1,Q))
        W_i = np.reshape(hyp[idx_3:idx_4], (Q,Q))

        # Update Cell State
        idx_1 = idx_4
        idx_2 = idx_1 + self.X_dim*Q
        idx_3 = idx_2 + Q
        idx_4 = idx_3 + Q*Q
        U_s = np.reshape(hyp[idx_1:idx_2], (self.X_dim,Q))
        b_s = np.reshape(hyp[idx_2:idx_3], (1,Q))
        W_s = np.reshape(hyp[idx_3:idx_4], (Q,Q))

        # Ouput Gate
        idx_1 = idx_4
        idx_2 = idx_1 + self.X_dim*Q
        idx_3 = idx_2 + Q
        idx_4 = idx_3 + Q*Q
        U_o = np.reshape(hyp[idx_1:idx_2], (self.X_dim,Q))
        b_o = np.reshape(hyp[idx_2:idx_3], (1,Q))
        W_o = np.reshape(hyp[idx_3:idx_4], (Q,Q))
                
        for i in range(0, self.lags):
            # Forget Gate
            F = self.sigmoid(np.matmul(H,W_f) + np.matmul(X[i,:,:],U_f) + b_f)
            # Input Gate
            I = self.sigmoid(np.matmul(H,W_i) + np.matmul(X[i,:,:],U_i) + b_i)
            # Update Cell State
            S_tilde = np.tanh(np.matmul(H,W_s) + np.matmul(X[i,:,:],U_s) + b_s)
            S = F*S + I*S_tilde
            # Ouput Gate
            O = self.sigmoid(np.matmul(H,W_o) + np.matmul(X[i,:,:],U_o) + b_o)
            H = O*np.tanh(S)
                
        idx_1 = idx_4
        idx_2 = idx_1 + Q*self.Y_dim
        idx_3 = idx_2 + self.Y_dim
        V = np.reshape(hyp[idx_1:idx_2], (Q,self.Y_dim))
        c = np.reshape(hyp[idx_2:idx_3], (1,self.Y_dim))
        Y = np.matmul(H,V) + c
        
        return Y
    
    
    # Evaluates the mean square error loss
    def loss(self, params):
        X = self.X_batch
        Y = self.Y_batch                          
        mu = self.forward_pass(X, params)                
        return np.mean((Y-mu)**2)
    
    
    
    # Fetches a mini-batch of data
    def fetch_minibatch_rnn(self, X,Y,N_batch):
        N = X.shape[1]
        idx = np.random.choice(N, N_batch, replace=False)
        X_batch = X[:,idx,:]
        Y_batch = Y[idx,:]
        return X_batch, Y_batch
    
    
    # Trains the model by minimizing the MSE loss
    def train(self, nIter = 10000, batch_size = 100):   

        start_time = timeit.default_timer()            
        for it in range(nIter):
            # Fetch minibatch
            self.X_batch, self.Y_batch = self.fetch_minibatch_rnn(self.X, self.Y, batch_size)
            
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
        y_star = self.forward_pass(X_star, self.params)
        return y_star
    
