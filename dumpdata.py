import sys

from mnist import TEST_FILE, load_data, print_row

if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else -1 
    test_data = load_data(TEST_FILE, delimiter=",", dtype=int)
    print(f"printing data for row {idx} from {TEST_FILE}")
    print_row(test_data[idx])