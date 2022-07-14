import numpy as np


def create_matrix(nrows, ncols):
    """Create a matrix with normally distributed random elements."""
    return np.random.default_rng().normal(loc=0, scale=1/(nrows*ncols), size=(nrows, ncols))


class Layer:
    """Model the connections between two sets of neurons in a network"""
    def __init__(self, ins, outs, act_function):
        self.ins = ins
        self.outs = outs
        self.act_function = act_function

        self._W = create_matrix(self.outs, self.ins)
        self._b = create_matrix(self.outs, 1)


    def y(self, inputs):
        """Computes and returns the pre-activation output of the layer"""
        return self._W @ inputs + self._b


    def output(self, inputs):
        """Compute this layer's outputs using the given inputs"""
        return self.act_function.f(self.y(inputs))

