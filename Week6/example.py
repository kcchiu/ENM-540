#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:26:56 2018

@author: Paris
"""

import numpy as np
import matplotlib.pyplot as plt

from models_numpy import RNN
#from models_pytorch import RNN

np.random.seed(1234)
    
if __name__ == "__main__":
    
    def create_dataset(data, lags):
        N = len(data)-lags
        data_dim = data.shape[1]
        X = np.zeros((lags, N, data_dim))
        Y = np.zeros((N, data_dim))
        for i in range(0,N):
            X[:,i,:] = data[i:(i+lags), :]
            Y[i,:] = data[i + lags, :]
        return X, Y
    
    # generate the dataset
    def f(t):
        f = np.sin(np.pi*t)
        return f
    
    
    t = np.arange(0,10,0.1)[:,None]
    dataset = f(t)
    
    # Use 2/3 of all data as training Data
    train_size = int(len(dataset) * (2.0/3.0))
    train = dataset[0:train_size,:]
    
    # reshape X and Y
    # X has the form lags x data x dim
    # Y has the form data x dim
    lags = 5
    X, Y = create_dataset(train, lags)
    
    # Model creation
    hidden_dim = 4
    model = RNN(X, Y, hidden_dim)
    
    model.train(nIter = 10000, batch_size = Y.shape[0])
    
    # Prediction
    pred = np.zeros((len(dataset)-lags, Y.shape[-1]))
    X_tmp =  np.copy(X[:,0:1,:])
    for i in range(0, len(dataset)-lags):
        pred[i] = model.predict(X_tmp)
        X_tmp[:-1,:,:] = X_tmp[1:,:,:] 
        X_tmp[-1,:,:] = pred[i]
        
    plt.figure(1)
    plt.plot(dataset[lags:], 'b-', linewidth = 2, label = "Exact")
    plt.plot(pred, 'r--', linewidth = 3, label = "Prediction")
    plt.plot(X.shape[1]*np.ones((2,1)), np.linspace(-1.75,1.75,2), 'k--', linewidth=2)
    plt.axis('tight')
    plt.xlabel('$t$')
    plt.ylabel('$y_t$')
    plt.legend(loc='lower left')
    
