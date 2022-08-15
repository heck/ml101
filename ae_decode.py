import sys
import numpy as np 

from neural_network import NeuralNetwork
from ae import TEST_FILE, load_data, decode, print_row

if __name__ == "__main__":
    # idx = int(sys.argv[1]) if len(sys.argv) > 1 else -1 
    # test_data = load_data(TEST_FILE, delimiter=",", dtype=int)
    # print(f"printing data for row {idx} from {TEST_FILE}")
    # print_row(test_data[idx])

    # digit = test_data[idx][0]

    ae_nn = NeuralNetwork.load("ae.pkl")

    # encoding = [[x] for x in range(10)]
    # encoding[digit] = [10]
    # output = decode(ae_nn, encoding) * 255
    # row = [digit] + output.flatten().tolist()
    # print_row(row)

    for digit in range(10):
        encoding = [[x] for x in range(10)]
        encoding[digit] = [10]
        output = decode(ae_nn, encoding) * 255
        row = [digit] + output.flatten().tolist()
        print_row(row)