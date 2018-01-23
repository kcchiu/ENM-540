import torch
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

if __name__ == "__main__": 
    
    # N is the number of training points.
    # D_in is input dimension
    # D_out is output dimension.
    N, D_in, D_out = 64, 1, 1
    
    # Create random input and output data
    X = lhs(D_in, N)
    y = 5*X + np.random.randn(N,D_out)
    
    # Create random Tensors to hold inputs and outputs, and wrap them in Variables.
    X = Variable(torch.from_numpy(X), requires_grad=False)
    y = Variable(torch.from_numpy(y), requires_grad=False)
    
    # Randomly initialize weights
    w = Variable(torch.randn(D_in, D_out).double(), requires_grad=True)
    
    learning_rate = 1e-3
    for it in range(1000):
      # Forward pass: compute predicted y
      y_pred = torch.matmul(X,w)
      
      # Compute and print loss
      loss = torch.sum((y_pred - y)**2)
      print("Iteration: %d, loss: %f" % (it, loss.data[0]))
      
      # Backprop 
      loss.backward()
      
      # Update weights
      w.data = w.data - learning_rate * w.grad.data
      
      # Reset gradient
      w.grad.data.zero_()
      
    y_pred = torch.matmul(X,w)
    
    plt.figure(1)
    plt.plot(X.data.numpy(),y.data.numpy(),'o')
    plt.plot(X.data.numpy(), y_pred.data.numpy())
    plt.show()


