import numpy as np

class LeakyReLU:
    """Leaky Rectified Linear Unit"""
    def __init__(self, leaky_param=0.1):
        self.alpha = leaky_param

    def f(self, x):
        return np.maximum(x, x*self.alpha)

    def df(self, x):
        """ the deriviative of the above function """
        return np.maximum(x > 0, self.alpha)


class MSELoss:
    """Mean Squared Error Loss function"""
    def loss(self, values, expected):
        return np.mean((values - expected)**2)

    def dloss(self, values, expected):
        """ the deriviative of the above function """
        return 2*(values - expected)/values.size