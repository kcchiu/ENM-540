#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:20:11 2018

@author: Paris
"""

import torch
import torch.utils.data
from torch.autograd import Variable, grad
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
    


# Define CNN architecture and forward pass
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.fc = torch.nn.Linear(7*7*32, 10)
        
    def forward_pass(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ConvNet:
    # Initialize the class
    def __init__(self, X, Y):  
        
        # Check if there is a GPU available
        if torch.cuda.is_available() == True:
            self.dtype_double = torch.cuda.FloatTensor
            self.dtype_int = torch.cuda.LongTensor
        else:
            self.dtype_double = torch.FloatTensor
            self.dtype_int = torch.LongTensor
        
        # Define PyTorch dataset
        X = torch.from_numpy(X).type(self.dtype_double) # num_images x num_pixels_x x num_pixels_y
        Y = torch.from_numpy(Y).type(self.dtype_int) # num_images x 1
        self.train_data = torch.utils.data.TensorDataset(X, Y)
        
        # Define architecture and initialize
        self.net = CNN()
        
        # Define the loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        
           
    # Trains the model by minimizing the Cross Entropy loss
    def train(self, num_epochs = 10, batch_size = 128):
        
        # Create a PyTorch data loader object
        self.trainloader = torch.utils.data.DataLoader(self.train_data, 
                                                  batch_size=batch_size, 
                                                  shuffle=True)
       
        start_time = timeit.default_timer()
        for epoch in range(num_epochs):
            for it, (images, labels) in enumerate(self.trainloader):
                images = Variable(images)
                labels = Variable(labels)
        
                # Reset gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.net.forward_pass(images)
                
                # Compute loss
                loss = self.loss_fn(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                self.optimizer.step()
        
                if (it+1) % 100 == 0:
                    elapsed = timeit.default_timer() - start_time
                    print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Time: %2fs' 
                           %(epoch+1, num_epochs, it+1, len(self.train_data)//batch_size, loss.data[0], elapsed))
                    start_time = timeit.default_timer()
                   
                    
    def test(self, X, Y):
        # Define PyTorch dataset
        X = torch.from_numpy(X).type(self.dtype_double) # num_images x num_pixels_x x num_pixels_y
        Y = torch.from_numpy(Y).type(self.dtype_int) # num_images x 1
        test_data = torch.utils.data.TensorDataset(X, Y)
       
        # Create a PyTorch data loader object
        test_loader = torch.utils.data.DataLoader(test_data, 
                                                  batch_size=128, 
                                                  shuffle=True)
        
        # Test prediction accuracy
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images)
            outputs = self.net.forward_pass(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('Test Accuracy of the model on the %d test images: %d %%' % (len(test_data), 100 * correct / total))
        
    
    # Evaluates predictions at test points    
    def predict(self, X_star):
        X_star = torch.from_numpy(X_star).type(self.dtype_double) 
        X_star = Variable(X_star, requires_grad=False)
        y_star = self.net.forward_pass(X_star)
        y_star = y_star.data.numpy()
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