#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:20:11 2018

@author: Paris
"""

import tensorflow as tf
import numpy as np
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

        # Initialize network weights and biases        
        self.weights, self.biases = self.initialize_NN(layers)
        
        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.X_tf = tf.placeholder(tf.float32, shape=(None, self.X.shape[1]))
        self.Y_tf = tf.placeholder(tf.float32, shape=(None, self.Y.shape[1]))
        
        # Evaluate prediction
        self.Y_pred = self.forward_pass(self.X_tf)
        
        # Evaluate loss
        self.loss = tf.losses.mean_squared_error(self.Y_tf, self.Y_pred)
        
        # Define optimizer        
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.train_op = self.optimizer.minimize(self.loss)
        
        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    
    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):      
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)   
        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
       
           
    # Evaluates the forward pass
    def forward_pass(self, H):
        num_layers = len(self.layers)
        for l in range(0,num_layers-2):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H
    
    
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
            # Fetch a mini-batch of data
            X_batch, Y_batch = self.fetch_minibatch(self.X, self.Y, batch_size)
            
            # Define a dictionary for associating placeholders with data
            tf_dict = {self.X_tf: X_batch, self.Y_tf: Y_batch}  
            
            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = timeit.default_timer()
                
                
    # Evaluates predictions at test points           
    def predict(self, X_star):      
        # Normalize inputs
        X_star = (X_star - self.Xmean) / self.Xstd
        tf_dict = {self.X_tf: X_star}       
        Y_star = self.sess.run(self.Y_pred, tf_dict) 
        # De-normalize outputs
        Y_star = Y_star * self.Ystd + self.Ymean
        return Y_star
    
    
    
class PDEnet:
    # Initialize the class
    def __init__(self, X_u, Y_u, X_f, Y_f, layers):
     
        self.X_u = X_u
        self.Y_u = Y_u
        
        self.X_f = X_f
        self.Y_f = Y_f
        
        self.layers = layers

        # Initialize network weights and biases        
        self.weights, self.biases = self.initialize_NN(layers)
        
        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.Xu_tf = tf.placeholder(tf.float32, shape=(None, self.X_u.shape[1]))
        self.Yu_tf = tf.placeholder(tf.float32, shape=(None, self.Y_u.shape[1]))
        self.Xf_tf = tf.placeholder(tf.float32, shape=(None, self.X_f.shape[1]))
        self.Yf_tf = tf.placeholder(tf.float32, shape=(None, self.Y_f.shape[1]))
        
        # Evaluate prediction
        self.u_pred = self.net_u(self.Xu_tf)
        self.f_pred = self.net_f(self.Xf_tf)
        
        # Evaluate loss
        self.loss = tf.losses.mean_squared_error(self.Yu_tf, self.u_pred) + \
                    tf.losses.mean_squared_error(self.Yf_tf, self.f_pred)
        
        # Define optimizer (use L-BFGS for better accuracy)       
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        
        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    
    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):      
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)   
        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
       
           
    # Evaluates the forward pass
    def forward_pass(self, H):
        num_layers = len(self.layers)
        for l in range(0,num_layers-2):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H
    
    
    # Forward pass for u
    def net_u(self, x):
        u = self.forward_pass(x)
        return u
    
    
    # Forward pass for f
    def net_f(self, x):
        u = self.net_u(x)
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_xx - u
        return f
    
    # Callback to print the loss at every optimization step
    def callback(self, loss):
        print('Loss:', loss)

       
    # Trains the model by minimizing the loss using L-BFGS
    def train(self): 
        
        # Define a dictionary for associating placeholders with data
        tf_dict = {self.Xu_tf: self.X_u, self.Yu_tf: self.Y_u,
                   self.Xf_tf: self.X_f, self.Yf_tf: self.Y_f}  
        
        # Call SciPy's L-BFGS otpimizer
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)

                
    # Evaluates predictions at test points           
    def predict_u(self, X_star):      
        tf_dict = {self.Xu_tf: X_star}       
        u_star = self.sess.run(self.u_pred, tf_dict) 
        return u_star
    
    # Evaluates predictions at test points           
    def predict_f(self, X_star):      
        tf_dict = {self.Xf_tf: X_star}       
        f_star = self.sess.run(self.f_pred, tf_dict) 
        return f_star
