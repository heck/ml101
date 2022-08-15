import pickle
class NeuralNetwork:
    """A series of connected, compatible layers."""
    def __init__(self, layers, loss_function, learning_rate):
        self._layers = layers
        self._loss_function = loss_function
        self._learning_rate = learning_rate
        self._layer_inputs = []

        # Check layer compatibility
        for (from_, to_) in zip(self._layers[:-1], self._layers[1:]):
            if from_.outs != to_.ins:
                raise ValueError(f"Layers should have compatible shapes, these are not: from=[{from_.ins}, {from_.outs}] to=[{to_.ins}, {to_.outs}]")

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def forward_pass(self, x):
        self._layer_inputs = []
        latest_output = x
        for layer in self._layers:
            self._layer_inputs.append(latest_output)
            latest_output = layer.output(inputs=latest_output)
        return latest_output

    def decode(self, encodings, decoder_idx):
        latest_output = encodings
        for layer in self._layers[decoder_idx:]:
            latest_output = layer.output(inputs=latest_output)
        return latest_output

    def train(self, x, t):
        """Train the network on input x and expected output t"""
        # step 1: do a forward pass using the given input data (x) (note: inputs to each layer saved in self._layer_inputs)
        latest_output = self.forward_pass(x)

        # NOTE: all vars named dFoo_dBar represent "change in foo over change in bar" or dFoo/dBar

        # step 2: compute the derivative of the loss (IOW, "rate of error") at the final (output) layer
        dErr_dOut = self._loss_function.dloss(latest_output, t)  # dloss is the derivative of the loss function

        # step 3: propagate the error at each layer's output backwards through said layer ("backpropagation")
        #         IOW, "learn" by changing each layer's weights and biases by the amount of error each contributed
        for layer, layer_input in zip(self._layers[::-1], self._layer_inputs[::-1]):  # -1 is the "step" (IOW, go backwards through the layers and results)
            y        = layer.y(layer_input)      # compute the *pre-activation* output (y) for this layer
            dOut_dY  = layer.act_function.df(y)  # note: df() is the derivative of the activation function
            dErr_dY  = dErr_dOut * dOut_dY
            # Math says: dErr_dW = dErr_dY * dY_dW 
            #       but: dY_dW   = layer_input (don't ask, it's math!)
            #        so: dErr_dW = dErr_dY * input
            dErr_dW  = dErr_dY @ layer_input.T

            # Before updating the weights, calculate the rate of error coming into this layer (from the previous one)
            dErr_dOut = layer._W.T @ dErr_dY

            # Update weights
            layer._W -= self._learning_rate * dErr_dW  # note: learning_rate is a scalar
            # Update biases
            # Math says: dErr_dB = dErr_dY * dY_dB
            #       but: dY_dB   = 1 (see "don't ask" above)
            #        so: dErr_dB = dErr_dY
            layer._b -= self._learning_rate * dErr_dY  # note: learning_rate is a scalar
