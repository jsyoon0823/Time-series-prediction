# Codebase for "Time-series prediction" with RNN, GRU, LSTM and Attention

Authors: Jinsung Yoon
Contact: jsyoon0823@gmail.com

This directory contains implementations of basic time-series prediction
using RNN, GRU, LSTM or Attention methods.
To run the pipeline, simply run python3 -m main_time_series_prediction.py.

## Stages of time-series prediction framework:

-   Load dataset (Google stocks data)
-   Train model:
    (1) RNN based: Simple RNN, GRU, LSTM
    (2) Attention based
-   Evaluate the performance: MAE or MSE metrics

### Command inputs:

-   train_rate: training data ratio
-   seq_len: sequence length
-   task: classification or regression
-   model_type: rnn, lstm, gru, or attention
-   h_dim: hidden state dimensions
-   n_layer: number of layers
-   batch_size: the number of samples in each mini-batch
-   epoch: the number of iterations
-   learning_rate: learning rates
-   metric_name: mse or mae

### Example command

```shell
$ python3 main_time_series_prediction.py 
--train_rate 0.8 --seq_len 7 --task regression --model_type lstm
--h_dim 10 --n_layer 3 --batch_size 32 --epoch 100 --learning_rate 0.01
--metric_name mae
```

### Outputs

-   MAE or MSE performance of trained model