"""Utility functions for time-series prediction.

Author: Jinsung Yoon
Contact: jsyoon0823@gmail.com
------------------------------------
(1) MinMaxScaler: MinMax normalizer
(2) performance: performance evaluator
(3) binary_cross_entropy_loss: loss for RNN on classification
(4) mse_loss: loss for RNN on regression
(5) rnn_sequential: Architecture
"""

# Necessary packages
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers


def MinMaxScaler(data):    
  """Normalizer (MinMax criteria).
  
  Args:
    - data: original data
    
  Returns:
    - norm_data: normalized data
  """    
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-8)
  return norm_data


def performance(test_y, test_y_hat, metric_name):
  """Evaluate predictive model performance.
  
  Args:
    - test_y: original testing labels
    - test_y_hat: prediction on testing data
    - metric_name: 'mse' or 'mae'
    
  Returns:
    - score: performance of the predictive model
  """  
  assert metric_name in ['mse', 'mae']
  
  if metric_name == 'mse':
    score = mean_squared_error(test_y, test_y_hat)
  elif metric_name == 'mae':
    score = mean_absolute_error(test_y, test_y_hat)
    
  score = np.round(score, 4)
    
  return score


def binary_cross_entropy_loss (y_true, y_pred):
  """User defined cross entropy loss.
  
  Args:
    - y_true: true labels
    - y_pred: predictions
    
  Returns:
    - loss: computed loss
  """
  # Exclude masked labels
  idx = tf.cast((y_true >= 0), float)
  # Cross entropy loss excluding masked labels
  loss = -(idx * y_true * tf.math.log(y_pred) + \
           idx * (1-y_true) * tf.math.log(1-y_pred))
  return loss


def mse_loss (y_true, y_pred):
  """User defined mean squared loss.
  
  Args:
    - y_true: true labels
    - y_pred: predictions
    
  Returns:
    - loss: computed loss
  """
  # Exclude masked labels
  idx = tf.cast((y_true >= 0), float)
  # Mean squared loss excluding masked labels
  loss = idx * tf.pow(y_true - y_pred, 2)
  return loss


def rnn_sequential (model, model_name, h_dim, return_seq):
  """Add one rnn layer in sequential model.
  
  Args:
    - model: sequential rnn model
    - model_name: rnn, lstm, or gru
    - h_dim: hidden state dimensions
    - return_seq: True or False
    
  Returns:
    - model: sequential rnn model
  """
  
  if model_name == 'rnn':
    model.add(layers.SimpleRNN(h_dim, return_sequences=return_seq))
  elif model_name == 'lstm':
    model.add(layers.LSTM(h_dim, return_sequences=return_seq))
  elif model_name == 'gru':
    model.add(layers.GRU(h_dim, return_sequences=return_seq))
    
  return model