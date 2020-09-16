import numpy as np

"""
DO NOT EDIT ANY PARTS OTHER THAN "EDIT HERE" !!! 

[Description]
__init__ - Initialize necessary variables for optimizer class
input   : gamma, epsilon
return  : X

update   - Update weight for one minibatch
input   : w - current weight, grad - gradient for w, lr - learning rate
return  : updated weight 
"""

class SGD:
    def __init__(self, gamma, epsilon):
        # ========================= EDIT HERE =========================
        pass

        # =============================================================

    def update(self, w, grad, lr):
        updated_weight = None
        # ========================= EDIT HERE =========================

        updated_weight = w - (lr * grad)

        # =============================================================
        return updated_weight

class Momentum:
    def __init__(self, gamma, epsilon):
        # ========================= EDIT HERE =========================

        self.gamma = gamma
        self.update_vector = None

        # =============================================================

    def update(self, w, grad, lr):
        updated_weight = None
        # ========================= EDIT HERE =========================

        if self.update_vector is None:
            self.update_vector = lr * grad
        else:
            self.update_vector = (self.gamma * self.update_vector) + (lr*grad)
        updated_weight = w - self.update_vector

        # =============================================================
        return updated_weight


class RMSProp:
    # ========================= EDIT HERE =========================
    def __init__(self, gamma, epsilon):
        # ========================= EDIT HERE =========================

        self.gamma = gamma
        self.epsilon = epsilon
        self.g_vector = None

        # =============================================================

    def update(self, w, grad, lr):
        updated_weight = None
        # ========================= EDIT HERE =========================

        if self.g_vector is None:
            self.g_vector = np.square(grad)
        else:
            self.g_vector = (self.gamma * self.g_vector) + ((1 - self.gamma) * np.square(grad))
        updated_weight = w - ((lr * grad) / np.sqrt(self.g_vector + self.epsilon))

        # =============================================================
        return updated_weight