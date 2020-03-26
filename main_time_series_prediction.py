"""Time-series prediction main function

Author: Jinsung Yoon
Contact: jsyoon0823@gmail.com
------------------------------------
(1) Load data
(2) Train model (RNN, GRU, LSTM, Attention)
(3) Evaluate the trained model
"""

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import warnings
warnings.filterwarnings("ignore")

from data_loader import data_loader
from basic_rnn_lstm_gru import GeneralRNN
from basic_attention import Attention
from utils import performance


def main (args):  
  """Time-series prediction main function.
  
  Args:
    - train_rate: training data ratio
    - seq_len: sequence length
    - task: classification or regression
    - model_type: rnn, lstm, gru, or attention
    - h_dim: hidden state dimensions
    - n_layer: number of layers
    - batch_size: the number of samples in each mini-batch
    - epoch: the number of iterations
    - learning_rate: learning rates
    - metric_name: mse or mae
  """
  # Load data
  train_x, train_y, test_x, test_y = data_loader(args.train_rate, 
                                                 args.seq_len)
  
  # Model traininig / testing
  model_parameters = {'task': args.task,
                      'model_type': args.model_type,
                      'h_dim': args.h_dim,
                      'n_layer': args.n_layer,
                      'batch_size': args.batch_size,
                      'epoch': args.epoch,
                      'learning_rate': args.learning_rate}
  
  if args.model_type in ['rnn','lstm','gru']:
    general_rnn = GeneralRNN(model_parameters)    
    general_rnn.fit(train_x, train_y)
    test_y_hat = general_rnn.predict(test_x)
  elif args.model_type == 'attention':
    basic_attention = Attention(model_parameters)    
    basic_attention.fit(train_x, train_y)
    test_y_hat = basic_attention.predict(test_x)
  
  # Evaluation
  result = performance(test_y, test_y_hat, args.metric_name)
  print('Performance (' + args.metric_name + '): ' + str(result))
  
    
##  
if __name__ == '__main__':
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_rate',
      help='training data ratio',
      default=0.8,
      type=str)
  parser.add_argument(
      '--seq_len',
      help='sequence length',
      default=7,
      type=int)
  parser.add_argument(
      '--model_type',
      choices=['rnn','gru','lstm','attention'],
      default='attention',
      type=str)
  parser.add_argument(
      '--h_dim',
      default=10,
      type=int)
  parser.add_argument(
      '--n_layer',
      default=3,
      type=int)
  parser.add_argument(
      '--batch_size',
      default=32,
      type=int)
  parser.add_argument(
      '--epoch',
      default=100,
      type=int)
  parser.add_argument(
      '--learning_rate',
      default=0.01,
      type=float)
  parser.add_argument(
      '--task',
      choices=['classification','regression'],
      default='regression',
      type=str)
  parser.add_argument(
      '--metric_name',
      choices=['mse','mae'],
      default='mae',
      type=str)
  
  args = parser.parse_args() 
  
  # Call main function  
  main(args)