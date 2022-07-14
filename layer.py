import numpy as np


def create_matrix(nrows, ncols):
    """Create a matrix with normally distributed random elements."""
    return np.random.default_rng().normal(loc=0, scale=1/(nrows*ncols), size=(nrows, ncols))


class Layer:
    """Model the connections between two sets of neurons in a network."""
    def __init__(self, ins, outs, act_function):
        self.ins = ins
        self.outs = outs
        self.act_function = act_function

        self._W = create_matrix(self.outs, self.ins)
        self._b = create_matrix(self.outs, 1)

    def forward_pass(self, x):
        """Compute the next set of neuron states with the given set of states."""
        y = self._W @ x + self._b
        return self.act_function.f(y)

