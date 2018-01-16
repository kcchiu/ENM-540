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
    
    # Randomly initialize weights
    W = np.random.randn(D_in, D_out)
    
    learning_rate = 1e-6
    for it in range(50000):
      # Forward pass: compute predicted y
      y_pred = np.matmul(x,W)
      
      # Compute and print loss
      loss = np.sum((y_pred - y)**2)
      print("Iteration: %d, loss: %f" % (it, loss))
      
      # Backprop to compute gradients of W with respect to loss
      grad_y_pred = 2.0 * (y_pred - y)
      grad_W = x.T.dot(grad_y_pred)
      
      # Update weights
      W = W - learning_rate * grad_W
      
    plt.figure(1)
    plt.plot(x,y,'o')
    plt.plot(x, y_pred)
    plt.show()
