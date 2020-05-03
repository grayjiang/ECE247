import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #
    C, H, W = input_dim
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros((1,num_filters))
    temp = (int)((H / 2) * (W / 2) * num_filters)
    self.params['W2'] = weight_scale * np.random.randn(temp, hidden_dim)
    self.params['b2'] = np.zeros((1,hidden_dim))
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros((1,num_classes))

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #

    out_conv, cache_conv = conv_relu_pool_forward(X, W1, b1,conv_param, pool_param)
    out_relu, cashe_relu = affine_relu_forward(out_conv, W2, b2)
    scores, cache_out = affine_forward(out_relu, W3, b3)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #
    loss, dout = softmax_loss(scores, y)

    # Add regularization
    loss += self.reg * 0.5 * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))

    dX3, dw3, db3 = affine_backward(dout, cache_out)
    dX2, dw2, db2 = affine_relu_backward(dX3, cashe_relu)
    N,C, H, W = X.shape
    HH = (int)(H/2)
    WW = (int)(W/2)
    
    dX2 = dX2.reshape(N,-1,HH,WW)
    dX1, dw1, db1 = conv_relu_pool_backward(dX2, cache_conv)

    grads['b3'] = db3 
    grads['b2'] = db2 
    grads['b1'] = db1 

    grads['W3'] = dw3 + self.reg * W3
    grads['W2'] = dw2 + self.reg * W2
    grads['W1'] = dw1 + self.reg * W1

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads


class MyConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  convX2 - pool - convX3 - pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #
    C, H, W = input_dim
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    
    self.params['W2'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
    self.params['b2'] = np.zeros(num_filters)
    
    self.params['W3'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
    self.params['b3'] = np.zeros(num_filters)
    
    self.params['W4'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
    self.params['b4'] = np.zeros(num_filters)
    
    self.params['W5'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
    self.params['b5'] = np.zeros(num_filters)
    
    temp = (int)((H / 4) * (W / 4) * num_filters)
    self.params['W6'] = weight_scale * np.random.randn(temp, hidden_dim)
    self.params['b6'] = np.zeros(hidden_dim)
    
    self.params['W7'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b7'] = np.zeros(num_classes)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']
    W7, b7 = self.params['W7'], self.params['b7']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #

    
    out_conv1, cache_conv1 = conv_forward_fast(X, W1, b1,conv_param)
    out_conv2, cache_conv2 = conv_forward_fast(out_conv1, W2, b2,conv_param)
    out_pool1, cache_pool1 = max_pool_forward_fast(out_conv2, pool_param)

    out_conv3, cache_conv3 = conv_forward_fast(out_pool1, W3, b3,conv_param)
    out_conv4, cache_conv4 = conv_forward_fast(out_conv3, W4, b4,conv_param)
    out_conv5, cache_conv5 = conv_forward_fast(out_conv4, W5, b5,conv_param)
    out_pool2, cache_pool2 = max_pool_forward_fast(out_conv5, pool_param)

    out_relu, cashe_relu = affine_relu_forward(out_pool2, W6, b6)
    scores, cache_out = affine_forward(out_relu, W7, b7)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #
    loss, dout = softmax_loss(scores, y)

    # Add regularization
    loss += self.reg * 0.5 * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W4 ** 2) + np.sum(W3 ** 2) + np.sum(W5 ** 2)+ np.sum(W6 ** 2)+ np.sum(W7 ** 2))
    
    dX7, dw7, db7 = affine_backward(dout, cache_out)
    dX6, dw6, db6 = affine_relu_backward(dX7, cashe_relu)
    N,C, H, W = X.shape
    HH = (int)(H/4)
    WW = (int)(W/4)
    dX6 = dX6.reshape(N,-1,HH,WW)
    dX_pool2 = max_pool_backward_fast(dX6, cache_pool2)
    dX5, dw5, db5 = conv_backward_fast(dX_pool2, cache_conv5)
    dX4, dw4, db4 = conv_backward_fast(dX5, cache_conv4)
    dX3, dw3, db3 = conv_backward_fast(dX4, cache_conv3)

    dX_pool1 = max_pool_backward_fast(dX3, cache_pool1)
    dX2, dw2, db2 = conv_backward_fast(dX_pool1, cache_conv2)
    dX1, dw1, db1 = conv_backward_fast(dX2, cache_conv1)

    grads['b7'] = db7
    grads['b6'] = db6
    grads['b5'] = db5
    grads['b4'] = db4 
    grads['b3'] = db3 
    grads['b2'] = db2 
    grads['b1'] = db1 
    
    grads['W7'] = dw7 + self.reg * W7
    grads['W6'] = dw6 + self.reg * W6
    grads['W5'] = dw5 + self.reg * W5
    grads['W4'] = dw4 + self.reg * W4
    grads['W3'] = dw3 + self.reg * W3
    grads['W2'] = dw2 + self.reg * W2
    grads['W1'] = dw1 + self.reg * W1

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads


