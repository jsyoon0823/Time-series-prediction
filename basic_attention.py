"""Basic Attention core functions for time-series prediction.

Author: Jinsung Yoon
Contact: jsyoon0823@gmail.com
"""

# Necessary packages
import tensorflow as tf
import numpy as np
import os
import shutil


class Attention():
  """Attention class.
  
  Attributes:
    - model_parameters:
      - task: classificiation or regression
      - h_dim: hidden state dimensions
      - batch_size: the number of samples in each mini-batch
      - epoch: the number of iterations
      - learning_rate: learning rate of training
  """
  def __init__(self, model_parameters):

    tf.compat.v1.reset_default_graph()
    self.task = model_parameters['task']
    self.h_dim = model_parameters['h_dim']
    self.batch_size = model_parameters['batch_size']
    self.epoch = model_parameters['epoch']
    self.learning_rate = model_parameters['learning_rate']
    
    self.save_file_directory = 'tmp/attention/'
    
    
  def process_batch_input_for_RNN(self, batch_input):
    """Function to convert batch input data to use scan ops of tensorflow.
    
    Args:
      - batch_input: original batch input
    
    Returns:
      - x: batch_input for RNN 
    """
    batch_input_ = tf.transpose(batch_input, perm=[2, 0, 1])
    x = tf.transpose(batch_input_)
    return x


  def sample_X(self, m, n):
    """Sample from the real data (Mini-batch index sampling).
    """
    return np.random.permutation(m)[:n]  
  
  
  def fit(self, x, y):
    """Train the model.
    
    Args:
      - x: training feature
      - y: training label
    """
        
    # Basic parameters
    no, seq_len, x_dim = x.shape
    y_dim = len(y[0, :])
    
    # Weights for GRU
    Wr = tf.Variable(tf.zeros([x_dim, self.h_dim]))
    Ur = tf.Variable(tf.zeros([self.h_dim, self.h_dim]))
    br = tf.Variable(tf.zeros([self.h_dim]))
        
    Wu = tf.Variable(tf.zeros([x_dim, self.h_dim]))
    Uu = tf.Variable(tf.zeros([self.h_dim, self.h_dim]))
    bu = tf.Variable(tf.zeros([self.h_dim]))
        
    Wh = tf.Variable(tf.zeros([x_dim, self.h_dim]))
    Uh = tf.Variable(tf.zeros([self.h_dim, self.h_dim]))
    bh = tf.Variable(tf.zeros([self.h_dim]))
                
    # Weights for attention mechanism 
    Wa1 = tf.Variable(tf.random.truncated_normal([self.h_dim + x_dim, 
                                                  self.h_dim], 
                                                 mean=0, stddev=.01))
    Wa2 = tf.Variable(tf.random.truncated_normal([self.h_dim, y_dim], 
                                                 mean=0, stddev=.01))
    ba1 = tf.Variable(tf.random.truncated_normal([self.h_dim], 
                                                 mean=0, stddev=.01))
    ba2 = tf.Variable(tf.random.truncated_normal([y_dim], mean=0, stddev=.01))
            
    # Weights for output layers
    Wo = tf.Variable(tf.random.truncated_normal([self.h_dim, y_dim], 
                                         mean=0, stddev=.01))
    bo = tf.Variable(tf.random.truncated_normal([y_dim], mean=0, stddev=.01))
    
    # Target
    Y = tf.compat.v1.placeholder(tf.float32, [None,1])    
    # Input vector with shape[batch, seq, embeddings]
    _inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, None, x_dim], 
                                       name='inputs')
    
    # Processing inputs to work with scan function
    processed_input = self.process_batch_input_for_RNN(_inputs)
            
    # Initial Hidden States
    initial_hidden = _inputs[:, 0, :]
    initial_hidden = tf.matmul(initial_hidden, tf.zeros([x_dim, self.h_dim]))
        
 
    def GRU(previous_hidden_state, x):
      """Function for Forward GRU cell.
      
      Args:
        - previous_hidden_state
        - x: current input
        
      Returns:
        - current_hidden_state
      """
      # R Gate
      r = tf.sigmoid(tf.matmul(x, Wr) + \
                     tf.matmul(previous_hidden_state, Ur) + br)
      # U Gate
      u = tf.sigmoid(tf.matmul(x, Wu) + \
                     tf.matmul(previous_hidden_state, Uu) + bu)
      # Final Memory cell
      c = tf.tanh(tf.matmul(x, Wh) + \
                  tf.matmul( tf.multiply(r, previous_hidden_state), Uh) + bh)
      # Current Hidden state
      current_hidden_state = tf.multiply( (1 - u), previous_hidden_state ) + \
                             tf.multiply( u, c )
      return current_hidden_state
        
    
    def get_states():
      """Function to get the hidden and memory cells after forward pass.
      
      Returns:
        - all_hidden_states
      """
      # Getting all hidden state through time
      all_hidden_states = tf.scan(GRU, processed_input, 
                                  initializer=initial_hidden, name='states')
      return all_hidden_states
              
        
    def get_attention(hidden_state):
      """Function to get attention with the last input.
      
      Args:
        - hidden_states
        
      Returns:
        - e_values
      """
      inputs = tf.concat((hidden_state, processed_input[-1]), axis = 1)
      hidden_values = tf.nn.tanh(tf.matmul(inputs, Wa1) + ba1)
      e_values = (tf.matmul(hidden_values, Wa2) + ba2)
      return e_values
        

    def get_outputs():
      """Function for getting output and attention coefficient.
      
      Returns:
        - output: final outputs
        - a_values: attention values
      """
      all_hidden_states = get_states()
      all_attention = tf.map_fn(get_attention, all_hidden_states)
      a_values = tf.nn.softmax(all_attention, axis = 0)
      final_hidden_state = tf.einsum('ijk,ijl->jkl', a_values, 
                                     all_hidden_states)
      output = tf.nn.sigmoid(tf.matmul(final_hidden_state[:,0,:], Wo) + bo, 
                             name='outputs')
      return output, a_values   
                
    # Getting all outputs from rnn
    outputs, attention_values = get_outputs()
    
    # reshape out for sequence_loss
    if self.task == 'classification':
      loss = tf.reduce_mean(Y * tf.log(outputs + 1e-8) + \
                            (1-Y) * tf.log(1-outputs + 1e-8))
    elif self.task == 'regression':
      loss = tf.sqrt(tf.reduce_mean(tf.square(outputs - Y)))
    
    # Optimization
    optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
    train = optimizer.minimize(loss)

    # Sessions
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
        
    # Training
    iteration_per_epoch = int(no/self.batch_size)
    iterations = int((self.epoch * no) / self.batch_size)
    
    for i in range(iterations):
      
      idx = self.sample_X(no, self.batch_size)
      Input = x[idx,:,:]            
      _, step_loss = sess.run([train, loss], 
                              feed_dict={Y: y[idx], _inputs: Input})
                
      # Print intermediate results
      if i % iteration_per_epoch == iteration_per_epoch-1:
        print('Epoch: ' + str(int(i/iteration_per_epoch)) + 
              ', Loss: ' + str(np.round(step_loss, 4)))
        
    # Reset the directory for saving
    if not os.path.exists(self.save_file_directory):
      os.makedirs(self.save_file_directory)
    else:
      shutil.rmtree(self.save_file_directory)
  
    # Save model
    inputs = {'inputs': _inputs}
    outputs = {'outputs': outputs}
    tf.compat.v1.saved_model.simple_save(sess, self.save_file_directory, 
                                         inputs, outputs)    
        
       
  def predict(self, test_x):
    """Prediction with trained model.
    
    Args:
      - test_x: testing features
      
    Returns:
      - test_y_hat: predictions on testing set
    """
    
    graph = tf.Graph()
    with graph.as_default():
      with tf.compat.v1.Session() as sess:
        tf.compat.v1.saved_model.loader.load(sess, [tf.saved_model.SERVING], 
                                             self.save_file_directory)
        x = graph.get_tensor_by_name('inputs:0')
        outputs = graph.get_tensor_by_name('outputs:0')
    
        test_y_hat = sess.run(outputs, feed_dict={x: test_x})
        
    return test_y_hat
    