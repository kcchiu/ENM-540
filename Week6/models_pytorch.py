#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:52:44 2018

@author: Paris
"""

import torch
from torch.autograd import Variable
import timeit


class RNN:
    # Initialize the class
    def __init__(self, X, Y, hidden_dim):        

        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        self.X = Variable(X, requires_grad=False)
        self.Y = Variable(Y, requires_grad=False)
        
        self.hidden_dim = hidden_dim
       
        self.net = torch.nn.RNN(X.shape[2], hidden_dim, 1)    
        self.linear = torch.nn.Linear(hidden_dim,Y.shape[1])
        
        self.loss_fn = torch.nn.MSELoss()
        
        self.optimizer = torch.optim.Adam([
                    {'params': self.net.parameters(), 'weight_decay': 0.0},
                    {'params': self.linear.parameters(), 'weight_decay': 0.0}
                    ], lr=1e-3)
    
        
    def forward_pass(self, X):
        h0 = Variable(torch.zeros(1, X.data.shape[1], self.hidden_dim), requires_grad=False)
        output, hn = self.net.forward(X, h0)
        return self.linear.forward(output[-1])
    
    
    # Fetches a mini-batch of data
    def fetch_minibatch_rnn(self, X,Y,N_batch):
        N = X.data.shape[1]
        idx = torch.randperm(N)[0:N_batch]
        X_batch = X[:,idx,:]
        Y_batch = Y[idx,:]
        return X_batch, Y_batch
    
       
    def train(self, nIter = 10000, batch_size = 100):  
        
        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch minibatch
            X_batch, Y_batch = self.fetch_minibatch_rnn(self.X, self.Y, batch_size)
            
            # Forward pass: compute predicted y by passing x to the model.
            Y_pred = self.forward_pass(X_batch)
        
            # Compute and print loss.
            loss = self.loss_fn(Y_pred, Y_batch)
            
            self.optimizer.zero_grad()
        
            # Backward pass
            loss.backward()
            
            # update parameters
            self.optimizer.step()
            
            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss.data.numpy(), elapsed))
                start_time = timeit.default_timer()
    
    
    def predict(self, X_star):
        X_star = torch.from_numpy(X_star).float()
        X_star = Variable(X_star, requires_grad=False)
        y_star = self.forward_pass(X_star)
        y_star = y_star.data.numpy()
        return y_star
    
