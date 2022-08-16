import sys
import numpy as np 

from neural_network import NeuralNetwork
from ae import LATENT_OUPUT_LAYER, TEST_FILE, load_data, decode, print_row, to_col, test_row

def encoding_to_string(encoding):
    # return "".join(f"{digit}:{x[0]:.4f} " for digit, x in enumerate(encoding))
    max_idx = np.argmax(encoding)
    _str = ""
    for idx, val in enumerate(encoding):
        _str += f" [{idx}]{val[0]:.4f}"
        if idx == max_idx:
            _str += "<-"
    return _str


if __name__ == "__main__":
    # get the target digit and print it
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else -1 
    test_data = load_data(TEST_FILE, delimiter=",", dtype=int)
    print(f"printing data for row {idx} from {TEST_FILE}")
    print_row(test_data[idx])

    ae_nn = NeuralNetwork.load("ae.pkl")

    # run the target digit through the autoencoder and print the result
    digit, data = test_data[idx][0], test_data[idx][1:]
    output = ae_nn.forward_pass(to_col(data)) * 255
    row = [digit] + output.flatten().tolist()
    print_row(row)
    encoding = ae_nn._layer_inputs[LATENT_OUPUT_LAYER]
    print(f"encoding={encoding_to_string(encoding)}")

    max_idx = np.argmax(encoding)
    norm_encoding = encoding/encoding[max_idx][0]
    output = decode(ae_nn, norm_encoding) * 255
    row = [digit] + output.flatten().tolist()
    print(f"\n\nnomalized encoding = {encoding_to_string(norm_encoding)}")
    print_row(row)

    print("\n-------- enumerated decreasing normalized decodings --------\n")
    iters = 10
    dnorm_encoding = norm_encoding / iters
    for iter in range(iters):
        norm_encoding = norm_encoding - dnorm_encoding
        norm_encoding[max_idx] = 1
        output = decode(ae_nn, norm_encoding) * 255
        row = [iter] + output.flatten().tolist()
        print(f"norm_encoding iter {iter} = {encoding_to_string(norm_encoding)}")
        print_row(row)

    # print("\n-------- enumerated decodings --------\n")
    # for digit in range(10):
    #     # encoding = np.zeros((10, 1)) + 3.5
    #     # encoding[digit] = 8 
    #     encoding = np.zeros((10, 1))
    #     encoding[digit] = 1 
    #     output = decode(ae_nn, encoding) * 255
    #     row = [digit] + output.flatten().tolist()
    #     print(f"encoding for digit {digit} = {encoding_to_string(encoding)}")
    #     print_row(row)