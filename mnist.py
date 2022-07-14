import pathlib

import csv
import numpy as np

from funcs import LeakyReLU, MSELoss
from layer import Layer
from neural_network import NeuralNetwork

TRAIN_FILE = f"{pathlib.Path(__file__).parent}/mnistdata/mnist_train.csv"
TEST_FILE = f"{pathlib.Path(__file__).parent}/mnistdata/mnist_test.csv"


def load_data(filepath, delimiter=",", dtype=float):
    """Load a numerical numpy array from a file."""

    print(f"Loading {filepath}...")
    with open(filepath, "r") as f:
        data_iterator = csv.reader(f, delimiter=delimiter)
        data_list = list(data_iterator)
    data = np.asarray(data_list, dtype=dtype)
    print("Done.")
    return data


def to_col(x):
    return x.reshape((x.size, 1))


def test(net, test_data):
    correct = 0
    for test_row in test_data:
        t = test_row[0]
        x = to_col(test_row[1:])
        out = net.forward_pass(x)
        guess = np.argmax(out)
        if t == guess:
            correct += 1

    return correct/test_data.shape[0]


def train(net, train_data, validate_data):
    # Make a dict that maps each target numeral to a 10-wide boolean vector where the
    # correct answer is a 1 and all others are 0 (IOW, 10 "one-hot" vectors of length 10)
    # e.g. ts[3] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    ts = {}
    for t in range(10):
        tv = np.zeros((10, 1))
        tv[t] = 1
        ts[t] = tv

    for i, train_row in enumerate(train_data):
        t = ts[train_row[0]]
        x = to_col(train_row[1:])
        net.train(x, t)

        if not i%1000 and i > 0:
            accuracy = test(net, validate_data)
            print(f"After training on {i} samples (of {len(train_data)}) accuracy is {100*accuracy:.2f}%")


if __name__ == "__main__":
    layers = [
        Layer(784, 16, LeakyReLU()),
        Layer(16, 16, LeakyReLU()),
        Layer(16, 10, LeakyReLU()),
    ]
    nn = NeuralNetwork(layers, MSELoss(), 0.001)

    test_data = load_data(TEST_FILE, delimiter=",", dtype=int)
    accuracy = test(nn, test_data)
    print(f"Without training accuracy is {100*accuracy:.2f}%")     # Expected to be around 10%

    train_data = load_data(TRAIN_FILE, delimiter=",", dtype=int)
    train(nn, train_data, validate_data=test_data)

    accuracy = test(nn, test_data)
    print(f"Done trainining, final accuracy is {100*accuracy:.2f}%")
