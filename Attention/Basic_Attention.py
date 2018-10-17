'''
Jinsung Yoon (10/17/2018)
Basic Attention for Time-series Prediction
'''

#%% Setup
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops

# Functions
import sys
sys.path.append('/home/vdslab/Documents/Jinsung/2019_Research/ICML/Reference_Code/Attention')

# 1. Data Loading
from Data_Loader import Data_Loader

# 2. Parameters
# train Parameters
train_rate = 0.8

# 3. Data Loading
trainX, trainY, testX, testY = Data_Loader(train_rate)

#%% Main Function
# 1. Graph Initialization
ops.reset_default_graph()
    
# 2. Parameters
seq_length = len(trainX[0,:,0])
input_size = len(trainX[0,0,:])
target_size = len(trainY[0,:])    

learning_rate = 0.01
iterations = 10000
hidden_layer_size = 10
batch_size = 64
    
# 3. Weights and Bias
Wr = tf.Variable(tf.zeros([input_size, hidden_layer_size]))
Ur = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
br = tf.Variable(tf.zeros([hidden_layer_size]))
    
Wu = tf.Variable(tf.zeros([input_size, hidden_layer_size]))
Uu = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
bu = tf.Variable(tf.zeros([hidden_layer_size]))
    
Wh = tf.Variable(tf.zeros([input_size, hidden_layer_size]))
Uh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
bh = tf.Variable(tf.zeros([hidden_layer_size]))
            
# Weights for Attention 
Wa1 = tf.Variable(tf.truncated_normal([hidden_layer_size + input_size, hidden_layer_size], mean=0, stddev=.01))
Wa2 = tf.Variable(tf.truncated_normal([hidden_layer_size, target_size], mean=0, stddev=.01))
ba1 = tf.Variable(tf.truncated_normal([hidden_layer_size], mean=0, stddev=.01))
ba2 = tf.Variable(tf.truncated_normal([target_size], mean=0, stddev=.01))
        
# Weights for output layers
Wo = tf.Variable(tf.truncated_normal([hidden_layer_size, target_size], mean=0, stddev=.01))
bo = tf.Variable(tf.truncated_normal([target_size], mean=0, stddev=.01))

# 4. Place holder
# Target
Y = tf.placeholder(tf.float32, [None,1])    
# Input vector with shape[batch, seq, embeddings]
_inputs = tf.placeholder(tf.float32, shape=[None, None, input_size], name='inputs')
             
# Function to convert batch input data to use scan ops of tensorflow.
def process_batch_input_for_RNN(batch_input):
    
    batch_input_ = tf.transpose(batch_input, perm=[2, 0, 1])
    X = tf.transpose(batch_input_)
    
    return X
  
# Processing inputs to work with scan function
processed_input = process_batch_input_for_RNN(_inputs)
        
# Initial Hidden States
initial_hidden = _inputs[:, 0, :]
initial_hidden = tf.matmul(initial_hidden, tf.zeros([input_size, hidden_layer_size]))
    
# 5. Function for Forward GRU cell.
def GRU(previous_hidden_state, x):
    # R Gate
    r = tf.sigmoid(tf.matmul(x, Wr) + tf.matmul(previous_hidden_state, Ur) + br)
    
    # U Gate
    u = tf.sigmoid(tf.matmul(x, Wu) + tf.matmul(previous_hidden_state, Uu) + bu)
    
    # Final Memory cell
    c = tf.tanh(tf.matmul(x, Wh) + tf.matmul( tf.multiply(r, previous_hidden_state), Uh) + bh)
    
    # Current Hidden state
    current_hidden_state = tf.multiply( (1 - u), previous_hidden_state ) + tf.multiply( u, c )
    
    return current_hidden_state
    
# 6. Function to get the hidden and memory cells after forward pass
def get_states():
    # Getting all hidden state through time
    all_hidden_states = tf.scan(GRU, processed_input, initializer=initial_hidden, name='states')
    
    return all_hidden_states
          
#%% Attention
    
# Function to get attention with the last input
def get_attention(hidden_state):
                        
    inputs = tf.concat((hidden_state, processed_input[-1]), axis = 1)
    hidden_values = tf.nn.tanh(tf.matmul(inputs, Wa1) + ba1)
    e_values = (tf.matmul(hidden_values, Wa2) + ba2)
        
    return e_values
    
# Function for getting output and attention coefficient
def get_outputs():
  
    all_hidden_states = get_states()
            
    all_attention = tf.map_fn(get_attention, all_hidden_states)
    
    a_values = tf.nn.softmax(all_attention, axis = 0)
    
    final_hidden_state = tf.einsum('ijk,ijl->jkl',a_values, all_hidden_states)
        
    output = tf.nn.sigmoid(tf.matmul(final_hidden_state[:,0,:], Wo) + bo)

    return output, a_values   
            
# Getting all outputs from rnn
outputs, attention_values = get_outputs()

# reshape out for sequence_loss
loss = tf.sqrt(tf.reduce_mean(tf.square(outputs - Y)))

# Optimization
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# Sessions
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 3. Sample from the real data (Mini-batch index sampling)
def sample_X(m, n):
  return np.random.permutation(m)[:n]  

# Training step
for i in range(iterations):

    idx = sample_X(len(trainX[:,0,0]), batch_size)
  
    Input = trainX[idx,:,:]            
            
    _, step_loss = sess.run([train, loss], feed_dict={Y: trainY[idx], _inputs: Input})
            
    if i % 100 == 0:
        print("[step: {}] loss: {}".format(i, step_loss))
       
#%% Evaluation
final_outputs, final_attention_values = sess.run([outputs, attention_values], feed_dict={_inputs: testX})

MSE = np.mean(np.abs(final_outputs - testY))
  
print(MSE)   
    
