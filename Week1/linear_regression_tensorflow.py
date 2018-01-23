import tensorflow as tf

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
    
    # Create placeholders to hold inputs and outputs.
    X_tf = tf.placeholder(tf.float32, shape=(None, D_in))
    y_tf = tf.placeholder(tf.float32, shape=(None, D_out))

    # Create Variables for the weights and initialize them with random data.
    # A TensorFlow Variable persists its value across executions of the graph.
    w = tf.Variable(tf.random_normal((D_in, D_out)))
    
    # Define loss
    y_pred = tf.matmul(X_tf,w)
    loss = tf.reduce_sum((y_tf - y_pred) ** 2)
    
    # Define gradients
    grad_w = tf.gradients(loss, w)[0]
    
    # Update the weights using gradient descent. To actually update the weights
    # we need to evaluate new_W when executing the graph. Note that
    # in TensorFlow the the act of updating the value of the weights is part of
    # the computational graph; in PyTorch this happens outside the computational
    # graph.
    learning_rate = 1e-3
    new_w = w.assign(w - learning_rate * grad_w)
    
    # Now we have built our computational graph, so we enter a TensorFlow session to
    # actually execute the graph.
    with tf.Session() as sess:
        # Run the graph once to initialize the Variables w1 and w2.
        sess.run(tf.global_variables_initializer())
    
        for it in range(1000):
            loss_value, _ = sess.run([loss, new_w], feed_dict={X_tf: X, y_tf: y})
            print("Iteration: %d, loss: %f" % (it, loss_value))
        
        y_pred_values = sess.run(y_pred, feed_dict={X_tf: X})
    
    plt.figure(1)
    plt.plot(X,y,'o')
    plt.plot(X, y_pred_values)
    plt.show()

   