import numpy as np
from sklearn import preprocessing

def read_data(train_file, test_file):
    # Read data from files and preprocess it
    tr_data = np.genfromtxt(train_file, delimiter=',', dtype=str)
    te_data = np.genfromtxt(test_file, delimiter=',', dtype=str)
    
    labels_train = tr_data[1:, -1].astype(int)
    labels_test = te_data[1:, -1].astype(int)
    
    x_train = tr_data[1:, 5:-1].astype(float)
    x_test = te_data[1:, 5:-1].astype(float)
    
    return x_train, labels_train, x_test, labels_test

def rolling_window(a, window):
    # Create rolling windows for sequence data
    b = []
    for i in range(len(a)-window+1):
        b.append([])
        for j in range(window):
            b[-1].append(a[j+i])
    return b

def scale_data(data):
    # Scale continuous features
    scalar = preprocessing.StandardScaler().fit(data)
    scaled_data = scalar.transform(data)
    return scaled_data.tolist()
