"""General RNN core functions for time-series prediction.

Author: Jinsung Yoon
Contact: jsyoon0823@gmail.com
"""

# Necessary packages
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint
from utils import binary_cross_entropy_loss, mse_loss, rnn_sequential


class GeneralRNN():
  """RNN predictive model to time-series.
  
  Attributes:
    - model_parameters:
      - task: classification or regression
      - model_type: 'rnn', 'lstm', or 'gru'
      - h_dim: hidden dimensions
      - n_layer: the number of layers
      - batch_size: the number of samples in each batch
      - epoch: the number of iteration epochs
      - learning_rate: the learning rate of model training
  """

  def __init__(self, model_parameters):

    self.task = model_parameters['task']
    self.model_type = model_parameters['model_type']
    self.h_dim = model_parameters['h_dim']
    self.n_layer = model_parameters['n_layer']
    self.batch_size = model_parameters['batch_size']
    self.epoch = model_parameters['epoch']
    self.learning_rate = model_parameters['learning_rate']
    
    assert self.model_type in ['rnn', 'lstm', 'gru']

    # Predictor model define
    self.predictor_model = None

    # Set path for model saving
    model_path = 'tmp'
    if not os.path.exists(model_path):
      os.makedirs(model_path)
    self.save_file_name = '{}'.format(model_path) + \
                          datetime.now().strftime('%H%M%S') + '.hdf5'
  

  def _build_model(self, x, y):
    """Construct the model using feature and label statistics.
    
    Args:
      - x: features
      - y: labels
      
    Returns:
      - model: predictor model
    """    
    # Parameters
    h_dim = self.h_dim
    n_layer = self.n_layer
    dim = len(x[0, 0, :])
    max_seq_len = len(x[0, :, 0])

    model = tf.keras.Sequential()
    model.add(layers.Masking(mask_value=0., input_shape=(max_seq_len, dim)))

    for _ in range(n_layer - 1):
      model = rnn_sequential(model, self.model_type, h_dim, return_seq=True)

    model = rnn_sequential(model, self.model_type, h_dim, 
                           return_seq=False)
    adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, 
                                    beta_1=0.9, beta_2=0.999, amsgrad=False)

    if self.task == 'classification':
      model.add(layers.Dense(y.shape[-1], activation='sigmoid'))
      model.compile(loss=binary_cross_entropy_loss, optimizer=adam)
      
    elif self.task == 'regression':
      model.add(layers.Dense(y.shape[-1], activation='linear'))
      model.compile(loss=mse_loss, optimizer=adam, metrics=['mse'])

    return model
  

  def fit(self, x, y):
    """Fit the predictor model.
    
    Args:
      - x: training features
      - y: training labels
      
    Returns:
      - self.predictor_model: trained predictor model
    """
    idx = np.random.permutation(len(x))
    train_idx = idx[:int(len(idx)*0.8)]
    valid_idx = idx[int(len(idx)*0.8):]
    
    train_x, train_y = x[train_idx], y[train_idx]
    valid_x, valid_y = x[valid_idx], y[valid_idx]
    
    self.predictor_model = self._build_model(train_x, train_y)

    # Callback for the best model saving
    save_best = ModelCheckpoint(self.save_file_name, monitor='val_loss',
                                mode='min', verbose=False,
                                save_best_only=True)

    # Train the model
    self.predictor_model.fit(train_x, train_y, 
                             batch_size=self.batch_size, epochs=self.epoch, 
                             validation_data=(valid_x, valid_y), 
                             callbacks=[save_best], verbose=True)

    self.predictor_model.load_weights(self.save_file_name)
    os.remove(self.save_file_name)

    return self.predictor_model
  
  
  def predict(self, test_x):
    """Return the temporal and feature importance.
    
    Args:
      - test_x: testing features
      
    Returns:
      - test_y_hat: predictions on testing set
    """
    test_y_hat = self.predictor_model.predict(test_x)
    return test_y_hat
