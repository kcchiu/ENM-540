#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:36:07 2018

@author: Paris
"""

import numpy as np
import cPickle, gzip
import matplotlib.pyplot as plt

from models_pytorch import ConvNet


if __name__ == "__main__": 
    
    def plot_random_sample(images, labels):
        idx = np.random.randint(images.shape[0])
        plt.figure(1)
        plt.clf()
        img = images[idx,0,:,:]
        plt.imshow(img, cmap=plt.get_cmap('gray_r'))
        print('This is a %d' % labels[idx])
        plt.show()

    # Load the dataset
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    # Training data
    N_train = train_set[0].shape[0]
    train_images = train_set[0].reshape([N_train, 1, 28, 28])
    train_labels = train_set[1]
    
    # Test data
    N_test = test_set[0].shape[0]
    test_images = test_set[0].reshape([N_test, 1, 28, 28])
    test_labels = test_set[1]
    
    # Check a few samples to make sure the data was loaded correctly
    # plot_random_sample(train_images, train_labels)
    
    # Define model
    model = ConvNet(train_images, train_labels)
    
    # Train
    model.train()
    
    # Evaluate test performance
    model.test(test_images, test_labels)
    
    # Predict
    predicted_labels = np.argmax(model.predict(test_images),1)
    
    # Plot a random prediction
    idx = 5452
    plt.figure(1)
    img = test_images[idx,0,:,:]
    plt.imshow(img, cmap=plt.get_cmap('gray_r'))
    print('Correct label: %d, Predicted label: %d' % (test_labels[idx], predicted_labels[idx]))
    plt.show()