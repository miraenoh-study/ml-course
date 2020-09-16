import numpy as np
from utils import _initialize, optimizer

DATA_NAME = 'Iris'
print('hello')
train_data, test_data, softmax_classifier, accuracy = _initialize(DATA_NAME)
train_x, train_y = train_data


num_data, num_features = train_x.shape
num_label = int(train_y.max()) + 1
print('# of Training data : %d' % num_data)
print('# of features : ', num_features)
print('# of labels : ', num_label)

model = softmax_classifier(num_features, num_label)

optim = optimizer('SGD', gamma=0.9, epsilon=0.1)
model.train(train_x,train_y,1,10,0.001,optim)
