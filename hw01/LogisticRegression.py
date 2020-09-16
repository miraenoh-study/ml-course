import numpy as np

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses

        # ========================= EDIT HERE ========================

        # W를 0~1중 랜덤 값으로 설정
        self.W = np.random.rand(self.num_features, 1)

        # for graph
        epoch_val = []


        for i in range(epochs):
            num_remain = x.shape[0]

            index = 0

            cur_batch_size = batch_size
            while num_remain > 0:
                if num_remain < cur_batch_size:
                    cur_batch_size = num_remain

                x_batch = x[index:index+cur_batch_size]
                y_batch = np.reshape(y[index:index+cur_batch_size], (-1, 1))

                # sigmoid function
                sigmoid = self._sigmoid(x_batch)

                # Loss and Cost
                loss = sigmoid - y_batch
                cost = (np.sum(np.square(loss))) / (2 * cur_batch_size)

                # Gradient
                gradient = np.matmul(x_batch.transpose(), loss) / cur_batch_size

                # Optimize
                self.W = optim.update(self.W, gradient, lr)

                index += cur_batch_size
                num_remain -= cur_batch_size

        final_loss = cost
        # ============================================================
        return final_loss

    def eval(self, x):
        threshold = 0.5
        pred = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # The model predicts the label as 1 if the probability is greater or equal to 'threshold'
        # Otherwise, it predicts as 0

        # ========================= EDIT HERE ========================

        sigmoid = self._sigmoid(x)
        pred = sigmoid

        for i in range(0, x.shape[0]):
            if sigmoid[i] >= 0.5:
                pred[i] = 1
            elif 0 <= sigmoid[i] < 0.5:
                pred[i] = 0
            else:
                print('Not in range!!')

        # ============================================================

        return pred

    def _sigmoid(self, x):
        sigmoid = None

        # Sigmoid Function
        # The function returns the sigmoid of 'x'

        # ========================= EDIT HERE ========================

        f_function = np.matmul(x, self.W)
        sigmoid = 1 / (1 + np.exp(-f_function))

        # ============================================================
        return sigmoid