class NeuralNetwork:
    """A series of connected, compatible layers."""
    def __init__(self, layers, loss_function, learning_rate):
        self._layers = layers
        self._loss_function = loss_function
        self._learning_rate = learning_rate

        # Check layer compatibility
        for (from_, to_) in zip(self._layers[:-1], self._layers[1:]):
            if from_.outs != to_.ins:
                raise ValueError("Layers should have compatible shapes.")

    def forward_pass(self, x):
        out = x
        for layer in self._layers:
            out = layer.forward_pass(out)
        return out

    # def loss(self, values, expected):
    #     return self._loss_function.loss(values, expected)

    def train(self, x, t):
        """Train the network on input x and expected output t"""
		# step 1: do a forward pass using the given input data (x) but save the intermediate values inputed to each layer
        layer_inputs = []
        latest_values = x
        for layer in self._layers:
            layer_inputs.append(latest_values)
            latest_values = layer.forward_pass(latest_values)

		# step 2: compute the derivative of the loss at the final (output) layer
        # initial dx = foward pass output - expected output, so loss_func'(dx) gives us dy/dx
        dx = self._loss_function.dloss(latest_values, t)  # dloss is the derivative of the loss function

		# step 3: propagate the error at the output backwards through each layer of the network ("backpropagation")
        for layer, layer_input in zip(self._layers[::-1], layer_inputs[::-1]):  # -1 is the "step" (IOW, go backwards through the layers and results)
            # Compute changes to weights and biases due to learning (derivatives)
            y  = layer._W @ layer_input + layer._b
            db = layer.act_function.df(y) * dx  # note: df() is derivative of the activation function
            dx = layer._W.T @ db
            dW = db @ layer_input.T

            # Update weights
            layer._W -= self._learning_rate * dW
            # Update biases
            layer._b -= self._learning_rate * db