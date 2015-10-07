import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding = 'latin1')
    return dict

x_train = None
y_train = []

for i in range(1, 6):
    data_dictionary = unpickle("cifar-10-batches-py/data_batch_%d" % i)
    if x_train == None:
        x_train = data_dictionary['data']
    else:
        x_train = np.vstack((x_train, data_dictionary['data']))
    y_train = y_train + data_dictionary['labels']
y_train = np.array(y_train)
print(len(x_train[0]))
x_train = x_train.reshape((len(x_train), 3, 32, 32))
print(len(x_train[0]))

test_data_dictionary = unpickle("cifar-10-batches-py/test_batch")
x_test = test_data_dictionary['data']
x_test = x_test.reshape(len(x_test), 3, 32, 32)
y_test = np.array(test_data_dictionary['labels'])
