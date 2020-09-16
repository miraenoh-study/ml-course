import numpy as np

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch

        # Training should be done for 'epochs' times with minibatch size of 'batch_size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses

        # ========================= EDIT HERE ========================


        w = self.W

        # W를 0~1중 랜덤 값으로 설정
        w = np.random.rand(self.num_features, 1)

        for i in range(epochs):

            num_remain = x.shape[0]

            index = 0

            cur_batch_size = batch_size
            while num_remain > 0:
                if num_remain < cur_batch_size:
                    cur_batch_size = num_remain

                x_batch = x[index:index+cur_batch_size]
                y_batch = np.reshape(y[index:index+cur_batch_size], (-1, 1))

                # Hypothesis
                hypothesis = np.matmul(x_batch, w)

                # Loss and Cost
                loss = hypothesis - y_batch
                cost = (np.sum(np.square(loss)))/(2*cur_batch_size)

                # Gradient
                # gradient = np.mean(loss)
                gradient = np.matmul(x_batch.transpose(), loss) / cur_batch_size

                # Optimize
                w = optim.update(w, gradient, lr)

                index += cur_batch_size
                num_remain -= cur_batch_size

        final_loss = cost
        self.W = w
        # ============================================================

        return final_loss

    def eval(self, x):
        pred = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'

        # ========================= EDIT HERE ========================

        pred = np.matmul(x, self.W)

        # ============================================================
        return pred
