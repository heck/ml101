import pathlib

import csv
import numpy as np
np.seterr(all='raise')

from funcs import LeakyReLU, Sigmoid, MSELoss
from layer import Layer
from neural_network import NeuralNetwork

TRAIN_FILE = f"{pathlib.Path(__file__).parent}/mnistdata/mnist_train.csv"
TEST_FILE = f"{pathlib.Path(__file__).parent}/mnistdata/mnist_test.csv"

INPUTS = 784
HIDDEN = 64
LATENT = 10

encoder = [
    Layer(INPUTS, HIDDEN, LeakyReLU()),
    Layer(HIDDEN, HIDDEN, LeakyReLU()),
    Layer(HIDDEN, LATENT, LeakyReLU()),
]
LATENT_OUPUT_LAYER = len(encoder)
decoder = [
    Layer(LATENT, HIDDEN, LeakyReLU()),
    Layer(HIDDEN, HIDDEN, LeakyReLU()),
    Layer(HIDDEN, INPUTS, Sigmoid()),
]
layers = encoder + decoder

# ae_nn = NeuralNetwork(layers, MSELoss(), 0.001)
ae_nn = NeuralNetwork(layers, MSELoss(), 0.1)

def load_data(filepath, delimiter=",", dtype=float):
    """Load a numerical numpy array from a file."""

    print(f"Loading {filepath}...")
    with open(filepath, "r") as f:
        data_iterator = csv.reader(f, delimiter=delimiter)
        data_list = list(data_iterator)
    data = np.asarray(data_list, dtype=dtype)
    print(f"Done. Loaded {len(data):,} samples.")
    return data


def to_col(x):
    return x.reshape((x.size, 1)) / 255.0


def print_row(data):
    print(f"The following digit is suppose to be a {data[0]}:")
    for row in range(28):
        if not sum(data[28*row:28*(row + 1)]):
            continue
        for col in range(28):
            idx = row*28 + col
            print(" ", end="") if data[1+idx] == 0 else \
            print(".", end="") if data[1+idx] < 64 else \
            print("*", end="") if data[1+idx] < 128 else \
            print("X", end="") if data[1+idx] < 196 else \
            print("@" if data[1+idx] else " ", end="")
        print()
    print()


def test_row(nn, data):
    out = nn.forward_pass(data)
    loss = nn._loss_function.loss(out, data)
    # return np.argmax(out)
    return loss


def test(nn, test_data):
    # correct = 0
    # for row in test_data:
    #     label, data = row[0], row[1:]
    #     guess = test_row(nn, to_col(data))
    #     if label == guess:
    #         correct += 1

    # return correct/test_data.shape[0]

    loss_tot = 0.0
    for row in test_data:
        _, data = row[0], row[1:]
        loss = test_row(nn, to_col(data))
        loss_tot += loss

    return loss_tot/test_data.shape[0]


def train(nn, train_data, validate_data):
    # # Make a dict that maps each target numeral to a 10-wide boolean vector where the
    # # correct answer is a 1 and all others are 0 (IOW, 10 "one-hot" vectors of length 10)
    # # e.g. ts[3] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # ts = {}
    # for digit in range(10):
    #     tv = np.zeros((10, 1))
    #     tv[digit] = 1
    #     ts[digit] = tv

    # now do the actual training
    loss_min = 1e10
    for batch in range(1, 100):
        loss_min_batch = 1e10
        for i, train_row in enumerate(train_data):
            _, data = train_row[0], train_row[1:]
            data_col = to_col(data)
            nn.train(data_col, data_col)

            # print progress every 1,000 training samples
            if not i%1000 and i > 0:
                loss_avg = test(nn, validate_data)
                report_str = f"{batch:3}: After training on {i} samples (out of {len(train_data)}) avg test loss={loss_avg:.6f} (min={loss_min:.6f})"
                loss_min = min(loss_min, loss_avg)
                loss_min_batch = min(loss_min_batch, loss_avg)
                if loss_min == loss_avg:
                    nn.save("ae.pkl")
                    report_str += " <---"
                print(report_str)

        if loss_min_batch > loss_min:
            print(f"No improvement after batch {batch}. Stopping.")
            return

def decode(nn, encoding):
    return nn.decode(encoding, LATENT_OUPUT_LAYER)

if __name__ == "__main__":
    test_data = load_data(TEST_FILE, delimiter=",", dtype=int)
    loss_avg = test(ae_nn, test_data)
    print(f"Without training, average test set loss is {loss_avg:.6f}")     # Expected to be around 10%

    print("---- BEFORE TRAINING ----")
    row = test_data[-1]
    print_row(row)
    print(f"SURVEY SAYS?: {test_row(ae_nn, to_col(row[1:]))}")
    print("-------------------------")

    train_data = load_data(TRAIN_FILE, delimiter=",", dtype=int)
    train(ae_nn, train_data, test_data[:250])

    loss_avg = test(ae_nn, test_data)
    print(f"Done trainining, final average test set loss is {loss_avg:.6f}")

    print("---- AFTER TRAINING ----")
    row = test_data[-1]
    print_row(row)
    print(f"SURVEY SAYS?: {test_row(ae_nn, to_col(row[1:]))}")
    print("-------------------------")