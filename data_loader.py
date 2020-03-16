"""Data loader.

Author: Jinsung Yoon
Contact: jsyoon0823@gmail.com
----------------------------------------
Loads Google stock dataset with MinMax normalization.
Reference: https://finance.yahoo.com/quote/GOOGL/history?p=GOOGL
"""

# Necessary Packages
import numpy as np
from utils import MinMaxScaler


def data_loader(train_rate = 0.8, seq_len = 7):
  """Loads Google stock data.
  
  Args:
    - train_rate: the ratio between training and testing sets
    - seq_len: sequence length
    
  Returns:
    - train_x: training feature
    - train_y: training labels
    - test_x: testing features
    - test_y: testing labels
  """
  
  # Load data
  ori_data = np.loadtxt('data/google.csv', delimiter=',', skiprows = 1)
  # Reverse the time order
  reverse_data = ori_data[::-1]
  # Normalization
  norm_data = MinMaxScaler(reverse_data)
    
  # Build dataset
  data_x = []
  data_y = []
  
  for i in range(0, len(norm_data[:,0]) - seq_len):
    # Previous seq_len data as features
    temp_x = norm_data[i:i + seq_len,:]
    # Values at next time point as labels
    temp_y = norm_data[i + seq_len, [-1]]
    data_x = data_x + [temp_x]
    data_y = data_y + [temp_y]
    
  data_x = np.asarray(data_x)
  data_y = np.asarray(data_y)
            
  # Train / test Division   
  idx = np.random.permutation(len(data_x))
  train_idx = idx[:int(train_rate * len(data_x))]
  test_idx = idx[int(train_rate * len(data_x)):]
        
  train_x, test_x = data_x[train_idx, :, :], data_x[test_idx, :, :]
  train_y, test_y = data_y[train_idx, :], data_y[test_idx, :]
    
  return train_x, train_y, test_x, test_y
