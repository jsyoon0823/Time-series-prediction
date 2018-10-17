'''
Jinsung Yoon (10/16/2018)
Data Loading
'''

#%% Necessary Packages
import numpy as np

#%% Google data loading

'''
1. train_rate: training / testing set ratio
'''

def Data_Loader(train_rate = 0.8):
    
    #%% Normalization
    def MinMaxScaler(data):        
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        return numerator / (denominator + 1e-8)

    #%% Data Preprocessing
    xy = np.loadtxt('/home/vdslab/Documents/Jinsung/2019_Research/ICML/Reference_Code/Attention/Data/GOOGLE.csv', delimiter = ",",skiprows = 1)
    xy = xy[::-1]
    xy = MinMaxScaler(xy)
    
    #%% Parameters
    seq_length = 7
    
    # Dataset build
    dataX = []
    dataY = []
    for i in range(0, len(xy[:,0]) - seq_length):
        _x = xy[i:i + seq_length,:]
        _y = xy[i + seq_length, [-1]]
        dataX.append(_x)
        dataY.append(_y)
            
                
    #%% Train / Test Division   
    train_size = int(len(dataX) * train_rate)
        
    trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
    trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataX)])
    
    return trainX, trainY, testX, testY

