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
    x = lhs(D_in, N)
    y = 5*x + np.random.randn(N,D_out)
    
    # Create random Tensors to hold inputs and outputs, and wrap them in Variables.
    x = Variable(torch.from_numpy(x), requires_grad=False)
    y = Variable(torch.from_numpy(y), requires_grad=False)
    
    # Randomly initialize weights
    W = Variable(torch.randn(D_in, D_out).double(), requires_grad=True)
    
    learning_rate = 1e-6
    for it in range(50000):
      # Forward pass: compute predicted y
      y_pred = torch.matmul(x,W)
      
      # Compute and print loss
      loss = torch.sum((y_pred - y)**2)
      print("Iteration: %d, loss: %f" % (it, loss.data[0]))
      
      # Backprop 
      loss.backward()
      
      # Update weights
      W.data = W.data - learning_rate * W.grad.data
      
      # Reset gradient
      W.grad.data.zero_()
      
    plt.figure(1)
    plt.plot(x.data.numpy(),y.data.numpy(),'o')
    plt.plot(x.data.numpy(), y_pred.data.numpy())


