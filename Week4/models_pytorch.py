#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:20:11 2018

@author: Paris
"""

import torch
from torch.autograd import Variable, grad
import numpy as np
import timeit

class NeuralNetwork:
    # Initialize the class
    def __init__(self, X, Y, layers):    
        
        # Check if there is a GPU available
        if torch.cuda.is_available() == True:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
        
        # Normalize the data
        self.Xmean, self.Xstd = X.mean(0), X.std(0)
        self.Ymean, self.Ystd = Y.mean(0), Y.std(0)
        X = (X - self.Xmean) / self.Xstd
        Y = (Y - self.Ymean) / self.Ystd

        # Define PyTorch variables
        X = torch.from_numpy(X).type(self.dtype)
        Y = torch.from_numpy(Y).type(self.dtype)
        self.X = Variable(X, requires_grad=False)
        self.Y = Variable(Y, requires_grad=False)
        
        # Initialize network weights and biases
        self.net = self.init_NN(layers) 
        
        # Define loss function
        self.loss_fn = torch.nn.MSELoss()
        
        # Define optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        
        
    # Define and initialize the neural network  
    def init_NN(self, Q):
        layers = []
        num_layers = len(Q)
        if num_layers < 2:
            net = torch.nn.Sequential()
        else:
            for i in range(0, num_layers-2):
                layers.append(torch.nn.Linear(Q[i],Q[i+1]))
                layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Linear(Q[-2],Q[-1]))
            net = torch.nn.Sequential(*layers)
        return net
    
    
    # Evaluates the forward pass
    def forward_pass(self, X):
        return self.net(X)
    
    
    # Fetches a mini-batch of data
    def fetch_minibatch(self,X, Y, N_batch):
        N = X.data.shape[0]
        idx = torch.randperm(N)[0:N_batch]
        X_batch = X[idx,:]
        Y_batch = Y[idx,:]        
        return X_batch, Y_batch
       
        
    # Trains the model by minimizing the MSE loss
    def train(self, nIter = 10000, batch_size = 100):
        
        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch mini-batch
            X_batch, Y_batch = self.fetch_minibatch(self.X, self.Y, batch_size)
            
            # Forward pass: compute predicted y by passing x to the model.
            Y_pred = self.forward_pass(X_batch)
        
            # Compute loss
            loss = self.loss_fn(Y_pred, Y_batch)
            
            # Backward pass
            loss.backward()
            
            # update parameters
            self.optimizer.step()
            
            # Reset gradients for next step
            self.optimizer.zero_grad()
            
            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss.data.numpy(), elapsed))
                start_time = timeit.default_timer()
    
    
    # Evaluates predictions at test points    
    def predict(self, X_star):
        # Normalize inputs
        X_star = (X_star - self.Xmean) / self.Xstd            
        X_star = torch.from_numpy(X_star).type(self.dtype)
        
        X_star = Variable(X_star, requires_grad=False)
        y_star = self.forward_pass(X_star)
        
        # De-normalize outputs
        y_star = y_star.data.numpy()
        y_star = y_star*self.Ystd + self.Ymean
            
        return y_star


class PDEnet:
    # Initialize the class
    def __init__(self, X_u, Y_u, X_f, Y_f, layers):
     
        # Convert to torch variables
        X_u = torch.from_numpy(X_u).float()
        Y_u = torch.from_numpy(Y_u).float()
        self.X_u = Variable(X_u, requires_grad=True)
        self.Y_u = Variable(Y_u, requires_grad=False)
        
        X_f = torch.from_numpy(X_f).float()
        Y_f = torch.from_numpy(Y_f).float()
        self.X_f = Variable(X_f, requires_grad=True)
        self.Y_f = Variable(Y_f, requires_grad=False)
        
        self.layers = layers

        # Initialize network weights and biases 
        self.net = self.init_NN(layers) 
        
        # Define loss function
        self.loss_fn = torch.nn.MSELoss()
        
        # Define optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        
    
    # Define and initialize the neural network  
    def init_NN(self, Q):
        layers = []
        num_layers = len(Q)
        if num_layers < 2:
            net = torch.nn.Sequential()
        else:
            for i in range(0, num_layers-2):
                layers.append(torch.nn.Linear(Q[i],Q[i+1]))
                layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Linear(Q[-2],Q[-1]))
            net = torch.nn.Sequential(*layers)
        return net
       
           
    # Evaluates the forward pass for u(x)
    def net_u(self, x):
        return self.net(x)
    
    
    # Evaluates the forward pass for f(x)
    def net_f(self, x):
        u = self.net_u(x)
        u_x = grad(u,x,torch.ones(x.data.shape),create_graph=True)[0]
        u_xx = grad(u_x,x,torch.ones(x.data.shape),create_graph=True)[0] 
        f = u_xx - u
        return f
    
    
    # Trains the model by minimizing the MSE loss
    def train(self, nIter = 10000):
        
        start_time = timeit.default_timer()
        for it in range(nIter):
            # Forward pass: compute predicted y by passing x to the model.
            u_pred = self.net_u(self.X_u)
            f_pred = self.net_f(self.X_f)
        
            # Compute loss
            loss = self.loss_fn(u_pred, self.Y_u) + self.loss_fn(f_pred, self.Y_f)
            
            # Backward pass
            loss.backward()
            
            # update parameters
            self.optimizer.step()
            
            # Reset gradients for next step
            self.optimizer.zero_grad()
            
            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss.data.numpy(), elapsed))
                start_time = timeit.default_timer()
    
    
    # Evaluates u(x) predictions at test points    
    def predict_u(self, X_star):
        X_star = torch.from_numpy(X_star).float()
        X_star = Variable(X_star, requires_grad=True)
        u_star = self.net_u(X_star)
        return u_star.data.numpy()
    
    # Evaluates f(x) predictions at test points    
    def predict_f(self, X_star):
        X_star = torch.from_numpy(X_star).float()
        X_star = Variable(X_star, requires_grad=True)
        f_star = self.net_f(X_star)
        return f_star.data.numpy()