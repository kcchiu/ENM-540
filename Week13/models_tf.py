#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:20:11 2018

@author: Paris
"""

import tensorflow as tf
import numpy as np
import timeit


class GAN:
    # Initialize the class
    def __init__(self, X, layers_G, layers_D):
        
        # Normalize data
        self.Xmean, self.Xstd = X.mean(0), X.std(0)
        X = (X - self.Xmean) / self.Xstd
     
        self.X = X
        
        self.X_dim = layers_G[-1]
        self.Z_dim = layers_G[0]
        
        self.layers_G = layers_G
        self.layers_D = layers_D

        # Initialize network weights and biases        
        self.weights_G, self.biases_G = self.initialize_NN(layers_G)
        self.weights_D, self.biases_D = self.initialize_NN(layers_D)
        
        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.X_tf = tf.placeholder(tf.float32, shape=(None, self.X_dim))
        self.Z_tf = tf.placeholder(tf.float32, shape=(None, self.Z_dim))
        
        # Evaluate loss
        self.loss_G = self.G_loss(self.Z_tf)
        self.loss_D = self.D_loss(self.X_tf, self.Z_tf)
                
        # Evaluate prediction
        self.sample = self.net_G(self.Z_tf)

        # Define optimizer        
        self.optimizer_G = tf.train.AdamOptimizer(1e-3)
        self.optimizer_D = tf.train.AdamOptimizer(1e-3)
        
        # Define train Ops
        self.train_op_G = self.optimizer_G.minimize(self.loss_G, 
                                                    var_list = [self.weights_G, self.biases_G])
                                                                    
        self.train_op_D = self.optimizer_D.minimize(self.loss_D, 
                                                    var_list = [self.weights_D, self.biases_D])
                
        
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
    def forward_pass(self, H, layers, weights, biases):
        num_layers = len(layers)
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H
    
    
    def net_G(self, Z):
        X = self.forward_pass(Z, 
                              self.layers_G,
                              self.weights_G,
                              self.biases_G)
        return X
    

    def net_D(self, X):
        D_logit = self.forward_pass(X, 
                              self.layers_D,
                              self.weights_D,
                              self.biases_D)
        return D_logit
    
    
    def D_loss(self, X, Z):
        G_sample = self.net_G(Z)
        
        D_real = tf.sigmoid(self.net_D(X))
        D_fake = tf.sigmoid(self.net_D(G_sample))
        
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
        
        D_loss = D_loss_real + D_loss_fake

        return D_loss
        
    
    def G_loss(self, Z):
        G_sample = self.net_G(Z)
        D_fake = tf.sigmoid(self.net_D(G_sample))
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
        return G_loss
        
    
    # Fetches a mini-batch of data
    def fetch_minibatch(self,X, N_batch):
        N = X.shape[0]
        idx = np.random.choice(N, N_batch, replace=False)
        X_batch = X[idx,:]
        return X_batch
    
    
    # Trains the model by minimizing the MSE loss
    def train(self, nIter = 10000, batch_size = 100): 

        start_time = timeit.default_timer()
        for it in range(nIter):     
            # Fetch a mini-batch of data
            X_batch = self.fetch_minibatch(self.X, batch_size)
            Z_batch = np.random.randn(batch_size, self.Z_dim)
            
            # Define a dictionary for associating placeholders with data
            tf_dict = {self.X_tf: X_batch, self.Z_tf: Z_batch}  
            
            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op_D, tf_dict)
            self.sess.run(self.train_op_D, tf_dict)
            self.sess.run(self.train_op_G, tf_dict)            
            
            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_D = self.sess.run(self.loss_D, tf_dict)
                loss_G = self.sess.run(self.loss_G, tf_dict)
                print('It: %d, G_loss: %.2e, D_loss: %.2e, Time: %.2f' % 
                      (it, loss_G, loss_D, elapsed))
                start_time = timeit.default_timer()
                
                
    def generate_samples(self, N_samples):
        Z = np.random.randn(N_samples, self.Z_dim)
        tf_dict = {self.Z_tf: Z}       
        X_star = self.sess.run(self.sample, tf_dict) 
        # De-normalize outputs
        X_star = X_star * self.Xstd + self.Xmean
        return X_star

    
    
