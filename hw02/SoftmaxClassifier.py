import numpy as np

class SoftmaxClassifier:
    def __init__(self, num_features, num_label):
        self.num_features = num_features
        self.num_label = num_label
        self.W = np.zeros((self.num_features, self.num_label))

    def train(self, x, y, epochs, batch_size, lr, optimizer):
        """
        N : # of training data
        D : # of features
        C : # of classes

        [INPUT]
        x : (N, D), input data (first column is bias for all data)
        y : (N, )
        epochs: (int) # of training epoch to execute
        batch_size : (int) # of minibatch size
        lr : (float), learning rate
        optimizer : (Python class) Optimizer

        [OUTPUT]
        final_loss : (float) loss of last training epoch

        [Functionality]
        Given training data, hyper-parameters and optimizer, execute training procedure.
        Training should be done in minibatch (not the whole data at a time)
        Procedure for one epoch is as follow:
        - For each minibatch
            - Compute probability of each class for data
            - Compute softmax loss
            - Compute gradient of weight
            - Update weight using optimizer
        * loss of one epoch = Mean of minibatch losses
        (minibatch losses = [0.5, 1.0, 1.0, 0.5] --> epoch loss = 0.75)

        """
        print('========== TRAINING START ==========')
        final_loss = None   # loss of final epoch
        num_data, num_feat = x.shape
        losses = []
        for epoch in range(1, epochs + 1):
            batch_losses = []   # list for storing minibatch losses

        # ========================= EDIT HERE ========================

            num_remain = num_data
            cur_batch_size = batch_size
            index = 0

            while num_remain > 0:
                if num_remain < cur_batch_size:
                    cur_batch_size = num_remain

                x_batch = x[index:index + cur_batch_size]
                y_batch = np.reshape(y[index:index+cur_batch_size], (-1, 1))

                # 1. softmax function
                score = np.matmul(x_batch, self.W)
                softmax = self._softmax(score)

                # 2. gradient
                gradient = self.compute_grad(x_batch, self.W, softmax, y_batch)

                # 3. optimize
                for j in range (0, self.num_label):
                    weight_class = np.zeros((self.num_features, 1))
                    grad_class = np.zeros((self.num_features, 1))
                    for i in range(0, self.num_features):
                        weight_class[i] = self.W[i][j]
                        grad_class[i] = gradient[i][j]
                    weight_class = optimizer.update(weight_class, grad_class, lr)
                    for i in range(0, self.num_features):
                        self.W[i][j] = weight_class[i]

                # 4. find loss
                score = np.matmul(x_batch, self.W)
                softmax = self._softmax(score)
                batch_losses.append(self.softmax_loss(softmax, y_batch))

                index += cur_batch_size
                num_remain -= cur_batch_size

        # ============================================================
            epoch_loss = sum(batch_losses) / len(batch_losses)  # epoch loss
            # print loss every 10 epoch
            if epoch % 10 == 0:
                print('Epoch %d : Loss = %.4f' % (epoch, epoch_loss))
            # store losses
            losses.append(epoch_loss)
        final_loss = losses[-1]

        return final_loss

    def eval(self, x):
        """

        [INPUT]
        x : (N, D), input data

        [OUTPUT]
        pred : (N, ), predicted label for N test data

        [Functionality]
        Given N test data, compute probability and make predictions for each data.
        """
        pred = None
        # ========================= EDIT HERE ========================

        num_data = x.shape[0]
        pred = np.zeros((num_data, 1))

        score = np.matmul(x, self.W)

        for i in range (0, num_data):
            label = 0
            max = score[i][label]

            for j in range (1, self.num_label):
                if score[i][j] > max:
                    label = j
                    max = score[i][j]

            pred[i] = label

        # ============================================================
        return pred

    def softmax_loss(self, prob, label):
        """
        N : # of minibatch data
        C : # of classes

        [INPUT]
        prob : (N, C), probability distribution over classes for N data
        label : (N, ), label for each data

        [OUTPUT]
        softmax_loss : scalar, softmax loss for N input

        [Functionality]
        Given probability and correct label, compute softmax loss for N minibatch data
        """
        softmax_loss = 0.0
        # ========================= EDIT HERE ========================

        num_data = label.shape[0]


        for i in range (0, num_data):
            softmax_loss += -(np.log(prob[i][label[i]]))
        softmax_loss /= num_data


        # ============================================================
        return softmax_loss

    def compute_grad(self, x, weight, prob, label):
        """
        N : # of minibatch data
        D : # of features
        C : # of classes

        [INPUT]
        x : (N, D), input data
        weight : (D, C), Weight matrix of classifier
        prob : (N, C), probability distribution over classes for N data
        label : (N, ), label for each data. (0 <= c < C for c in label)

        [OUTPUT]
        gradient of weight: (D, C), Gradient of weight to be applied (dL/dW)

        [Functionality]
        Given input (x), weight, probability and label, compute gradient of weight.
        """
        grad_weight = np.zeros_like(weight, dtype=np.float32) # (D, C)
        # ========================= EDIT HERE ========================

        num_data = x.shape[0]

        # set gradient for each classes
        for j in range (0, self.num_label):
            y_binary = np.zeros((num_data, 1))
            hypothesis = np.zeros((num_data, 1))
            for i in range(0, num_data):
                # set binary y
                if label[i] == j:
                    y_binary[i] = 1
                else:
                    y_binary[i] = 0
                # set hypothesis (log of softmax)
                hypothesis[i] = np.log(prob[i][j])

            # loss function for current label
            loss = hypothesis - y_binary

            # gradient for current label
            grad_class = np.matmul(x.transpose(), loss)

            # put the gradient for current label to gradient weight
            for i in range(0, self.num_features):
                grad_weight[i][j] = grad_class[i]

        # ============================================================
        return grad_weight


    def _softmax(self, x):
        """
        [INPUT]
        x : (N, C), score before softmax

        [OUTPUT]
        softmax : (same shape with x), softmax distribution over axis-1

        [Functionality]
        Given an input x, apply softmax function over axis-1 (classes).
        """
        softmax = None
        # ========================= EDIT HERE ========================

        num_data, num_class = x.shape
        softmax = np.zeros((num_data, num_class))

        for i in range (0, num_data):
            for j in range (0, num_class):
                x[i][j] = np.exp(x[i][j])

        for i in range (0, num_data):
            sum = np.sum(x[i])
            for j in range (0, num_class):
                softmax[i][j] = x[i][j] / sum


        # ============================================================
        return softmax