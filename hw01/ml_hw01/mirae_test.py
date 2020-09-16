import numpy as np
from utils import _initialize, optimizer

DATA_NAME = 'Digit'

train_data, test_data, softmax_classifier, accuracy = _initialize(DATA_NAME)
train_x, train_y = train_data

num_data, num_features = train_x.shape
num_label = int(train_y.max()) + 1

print('num_data : ', num_data)
print('num_features : ', num_features)
model = logistic_regression(num_features)
loss = model.train(train_x, train_y, 100, 10, 0.01, optimizer('SGD', gamma=1, epsilon=1))
print(loss)
pred = model.eval(train_x)