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

class Sigmoid:
    def f(self, x):
        return 1/(1 + np.exp(-x))

    def df(self, x):
        return self.f(x) * (1 - self.f(x))

class SoftMax:
    def f(self, x):
        return np.exp(x)/np.sum(np.exp(x))

    def df(self, x):
        return self.f(x) * (1 - self.f(x))

# # a bounus loss function to play with!
# class CrossEntropyLoss:
#     """Cross entropy loss function following the pytorch docs."""
#     def loss(self, values, target_class):
#         return -values[target_class, 0] + np.log(np.sum(np.exp(values)))

#     def dloss(self, values, target_class):
#         d = np.exp(values)/np.sum(np.exp(values))
#         _target_class = target_class.astype(int)
#         d[_target_class, 0] -= 1
#         return d